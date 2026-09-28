"""Failure-state tests for AsyncMinHashLSH with a fake in-memory async storage.

No external service is required: the async storage factories in
``datasketch.aio.lsh`` are replaced with fakes that record the order of
mutating operations and inject a failure at the Nth bucket write or delete.

The invariant under test is the same as for the synchronous
``MinHashLSH``: after any internal failure, a key must not be left in only
one direction of the index (key -> Hs record without bucket entries, or
bucket entries without the key record). The guarantees checked here come
from the LSH layer itself, not from any backend-specific transaction
support.
"""

import pytest

from datasketch.aio.lsh import AsyncMinHashLSH
from datasketch.minhash import MinHash


class InjectedStorageError(RuntimeError):
    """Raised by the fake storage when the configured Nth operation is reached."""


class AsyncStorageProbe:
    """Records mutating storage ops in order and fails the Nth matching call."""

    def __init__(self):
        self.ops = []
        self.fail_tag = None
        self.fail_method = None
        self.fail_at = None
        self._count = 0

    def arm(self, tag, method, at):
        self.fail_tag = tag
        self.fail_method = method
        self.fail_at = at
        self._count = 0

    def record(self, tag, method):
        self.ops.append((tag, method))
        if tag == self.fail_tag and method == self.fail_method:
            self._count += 1
            if self._count == self.fail_at:
                raise InjectedStorageError(f"injected failure at {tag}.{method} call #{self._count}")


class FakeAsyncOrderedStorage:
    """In-memory stand-in for an async ordered storage.

    Unlike the synchronous dict backend, this fake honors ``buffer=True`` by
    deferring writes to ``empty_buffer`` so the session commit boundary is
    observable.
    """

    def __init__(self, probe, tag):
        self._data = {}
        self._buffered = []
        self._probe = probe
        self._tag = tag
        self.batch_size = 10000

    def _apply_insert(self, key, vals):
        self._data.setdefault(key, []).extend(vals)

    def _apply_remove(self, keys):
        for key in keys:
            self._data.pop(key, None)

    def _apply_remove_val(self, key, val):
        vals = self._data.get(key, [])
        if val not in vals:
            raise ValueError("val not present")
        vals.remove(val)

    async def insert(self, key, *vals, buffer=False):
        self._probe.record(self._tag, "insert")
        if buffer:
            self._buffered.append(("insert", key, vals))
        else:
            self._apply_insert(key, vals)

    async def get(self, key):
        return list(self._data.get(key, []))

    async def getmany(self, *keys):
        return [await self.get(key) for key in keys]

    async def has_key(self, key):
        return key in self._data

    async def keys(self):
        return list(self._data)

    async def remove(self, *keys, buffer=False):
        self._probe.record(self._tag, "remove")
        if buffer:
            self._buffered.append(("remove", keys))
        else:
            self._apply_remove(keys)

    async def remove_val(self, key, val, buffer=False):
        self._probe.record(self._tag, "remove_val")
        if buffer:
            self._buffered.append(("remove_val", key, val))
        else:
            self._apply_remove_val(key, val)

    async def size(self):
        return len(self._data)

    async def itemcounts(self):
        return {key: len(vals) for key, vals in self._data.items()}

    async def status(self):
        return {"keyspace_size": len(self._data)}

    async def empty_buffer(self):
        buffered, self._buffered = self._buffered, []
        for op in buffered:
            if op[0] == "insert":
                self._apply_insert(op[1], op[2])
            elif op[0] == "remove":
                self._apply_remove(op[1])
            else:
                self._apply_remove_val(op[1], op[2])

    async def close(self):
        pass


class FakeAsyncUnorderedStorage(FakeAsyncOrderedStorage):
    """In-memory stand-in for an async unordered (set-valued) storage."""

    def _apply_insert(self, key, vals):
        self._data.setdefault(key, set()).update(vals)

    def _apply_remove_val(self, key, val):
        vals = self._data.get(key, set())
        if val not in vals:
            raise ValueError("val not present")
        vals.remove(val)

    async def get(self, key):
        return frozenset(self._data.get(key, set()))


@pytest.fixture
def probe():
    return AsyncStorageProbe()


@pytest.fixture
async def lsh(probe, monkeypatch):
    async def make_ordered(config, name=None):
        return FakeAsyncOrderedStorage(probe, "keys")

    async def make_unordered(config, name=None):
        return FakeAsyncUnorderedStorage(probe, "bucket")

    monkeypatch.setattr("datasketch.aio.lsh.async_ordered_storage", make_ordered)
    monkeypatch.setattr("datasketch.aio.lsh.async_unordered_storage", make_unordered)
    return await AsyncMinHashLSH(threshold=0.5, num_perm=16, prepickle=False, storage_config={"type": "aiomongo"})


def _minhash(content, num_perm=16):
    m = MinHash(num_perm)
    m.update(content)
    return m


async def _assert_present_both_directions(index, key):
    stored_Hs = list(await index.keys.get(key))
    assert len(stored_Hs) == index.b, "key -> Hs record must hold one H per band"
    for H, table in zip(stored_Hs, index.hashtables):
        assert key in await table.get(H), "bucket entry missing for band"


async def _assert_absent_both_directions(index, key):
    assert list(await index.keys.get(key)) == [], "key -> Hs record must be gone"
    for table in index.hashtables:
        for H in await table.keys():
            assert key not in await table.get(H), "bucket entry left behind"


async def test_insert_query_remove_normal_completion(lsh, probe):
    m = _minhash(b"a")
    await lsh.insert(b"a", m)
    assert probe.ops[0] == ("keys", "insert")
    assert probe.ops[1:] == [("bucket", "insert")] * lsh.b
    await _assert_present_both_directions(lsh, b"a")
    assert b"a" in await lsh.query(m)

    await lsh.remove(b"a")
    assert probe.ops.count(("bucket", "remove_val")) == lsh.b
    assert probe.ops[-1] == ("keys", "remove")
    await _assert_absent_both_directions(lsh, b"a")
    assert b"a" not in await lsh.query(m)
    assert await lsh.is_empty()


async def test_insert_duplicate_key_raises_before_touching_storage(lsh, probe):
    m = _minhash(b"a")
    await lsh.insert(b"a", m)
    ops_before = list(probe.ops)
    with pytest.raises(ValueError):
        await lsh.insert(b"a", m)
    assert probe.ops == ops_before, "duplicate check must reject before any storage write"
    await _assert_present_both_directions(lsh, b"a")


@pytest.mark.parametrize("fail_at", [1, 3, 5])
async def test_insert_failure_at_nth_bucket_write_leaves_no_one_directional_residue(lsh, probe, fail_at):
    m = _minhash(b"x")
    await lsh.insert(b"ok", _minhash(b"ok"))
    probe.arm("bucket", "insert", fail_at)
    with pytest.raises(InjectedStorageError):
        await lsh.insert(b"x", m)
    # The failed key must not survive in only one direction.
    await _assert_absent_both_directions(lsh, b"x")
    assert b"x" not in await lsh.query(m)
    # Concurrent band writes are all settled before the raise, and every
    # band is compensated because the landed subset is unknowable.
    assert probe.ops.count(("bucket", "remove_val")) == lsh.b
    assert probe.ops[-1] == ("keys", "remove")
    # The pre-existing key is untouched and the blocked key can be retried.
    await _assert_present_both_directions(lsh, b"ok")
    await lsh.insert(b"x", m)
    await _assert_present_both_directions(lsh, b"x")
    assert b"x" in await lsh.query(m)


async def test_insert_failure_rollback_preserves_colliding_key(lsh, probe):
    m = _minhash(b"shared")
    await lsh.insert(b"good", m)  # same minhash -> shares every bucket with b"bad"
    probe.arm("bucket", "insert", 3)
    with pytest.raises(InjectedStorageError):
        await lsh.insert(b"bad", m)
    # Compensating the failed key must not evict the colliding good key.
    await _assert_present_both_directions(lsh, b"good")
    assert b"good" in await lsh.query(m)
    await _assert_absent_both_directions(lsh, b"bad")


@pytest.mark.parametrize("fail_at", [1, 3, 5])
async def test_remove_failure_at_nth_bucket_delete_restores_both_directions(lsh, probe, fail_at):
    m = _minhash(b"x")
    await lsh.insert(b"x", m)
    probe.arm("bucket", "remove_val", fail_at)
    with pytest.raises(InjectedStorageError):
        await lsh.remove(b"x")
    # The blocked remove must leave the key fully indexed, not stranded in
    # only one direction.
    await _assert_present_both_directions(lsh, b"x")
    assert b"x" in await lsh.query(m)
    # A retry after the blocked remove completes the cleanup.
    await lsh.remove(b"x")
    await _assert_absent_both_directions(lsh, b"x")
    assert b"x" not in await lsh.query(m)
    assert await lsh.is_empty()


async def test_remove_failure_at_key_record_delete_restores_buckets(lsh, probe):
    m = _minhash(b"x")
    await lsh.insert(b"x", m)
    probe.arm("keys", "remove", 1)
    with pytest.raises(InjectedStorageError):
        await lsh.remove(b"x")
    await _assert_present_both_directions(lsh, b"x")
    assert b"x" in await lsh.query(m)
    await lsh.remove(b"x")
    await _assert_absent_both_directions(lsh, b"x")


async def test_remove_missing_key_raises_without_touching_storage(lsh, probe):
    await lsh.insert(b"a", _minhash(b"a"))
    ops_before = list(probe.ops)
    with pytest.raises(ValueError):
        await lsh.remove(b"missing")
    assert probe.ops == ops_before, "missing-key check must reject before any storage write"
    await _assert_present_both_directions(lsh, b"a")


async def test_insertion_session_commits_only_at_close(lsh, probe):
    # The fake honors buffer=True, so buffered writes stay invisible until
    # the session is closed; an unclosed session leaves nothing behind
    # because nothing was applied yet.
    m = _minhash(b"a")
    session = lsh.insertion_session()
    await session.insert(b"a", m)
    assert b"a" not in await lsh.query(m)
    assert not await lsh.has_key(b"a")
    await session.close()
    assert b"a" in await lsh.query(m)
    await _assert_present_both_directions(lsh, b"a")
