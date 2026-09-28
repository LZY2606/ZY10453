import pickle
import unittest
from unittest.mock import patch

import mockredis
import numpy as np

from datasketch.lsh import MinHashLSH
from datasketch.minhash import MinHash
from datasketch.weighted_minhash import WeightedMinHashGenerator


def fake_redis(**kwargs):
    redis = mockredis.mock_redis_client(**kwargs)
    redis.connection_pool = None
    redis.response_callbacks = None
    return redis


def first_band_hash(data):
    return data


def second_band_hash(data):
    return data[::-1]


class TestMinHashLSH(unittest.TestCase):
    def test_init(self):
        lsh = MinHashLSH(threshold=0.8)
        self.assertTrue(lsh.is_empty())
        b1, r1 = lsh.b, lsh.r
        lsh = MinHashLSH(threshold=0.8, weights=(0.2, 0.8))
        b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)

    def test__H(self):
        """Check _H output consistent bytes length given
        the same concatenated hash value size.
        """
        for _l in range(2, 128 + 1, 16):
            lsh = MinHashLSH(num_perm=128)
            m = MinHash()
            m.update(b"abcdefg")
            m.update(b"1234567")
            lsh.insert("m", m)
            sizes = [len(H) for ht in lsh.hashtables for H in ht]
            self.assertTrue(all(sizes[0] == s for s in sizes))

    def test_unpacking(self):
        for b in range(2, 1024 + 1):
            lsh = MinHashLSH(num_perm=b * 4, params=(b, 4))
            m = MinHash(num_perm=b * 4)
            m.update(b"abcdefg")
            m.update(b"1234567")
            lsh.insert("m", m)
            sizes = [len(H) for ht in lsh.hashtables for H in ht]
            self.assertTrue(all(sizes[0] == s for s in sizes))

    def test_insert(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        for t in lsh.hashtables:
            self.assertTrue(len(t) >= 1)
            items = []
            for H in t:
                items.extend(t[H])
            self.assertTrue("a" in items)
            self.assertTrue("b" in items)
        self.assertTrue("a" in lsh)
        self.assertTrue("b" in lsh)
        for i, H in enumerate(lsh.keys["a"]):
            self.assertTrue("a" in lsh.hashtables[i][H])

        m3 = MinHash(18)
        self.assertRaises(ValueError, lsh.insert, "c", m3)

    def test_query(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        result = lsh.query(m1)
        self.assertTrue("a" in result)
        result = lsh.query(m2)
        self.assertTrue("b" in result)

        m3 = MinHash(18)
        self.assertRaises(ValueError, lsh.query, m3)

    def test_query_buffer(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        lsh.add_to_query_buffer(m1)
        result = lsh.collect_query_buffer()
        self.assertTrue("a" in result)
        lsh.add_to_query_buffer(m2)
        result = lsh.collect_query_buffer()
        self.assertTrue("b" in result)
        m3 = MinHash(18)
        self.assertRaises(ValueError, lsh.add_to_query_buffer, m3)

    def test_query_buffer_matches_query_candidates(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=32)
        docs = []
        for tokens in ([b"a", b"b", b"c"], [b"a", b"b", b"d"], [b"x", b"y", b"z"]):
            minhash = MinHash(num_perm=32)
            for token in tokens:
                minhash.update(token)
            docs.append(minhash)
        for key, minhash in enumerate(docs):
            lsh.insert(key, minhash)

        lsh.add_to_query_buffer(docs[0])
        buffered_result = set(lsh.collect_query_buffer())
        direct_result = set(lsh.query(docs[0]))

        self.assertEqual(buffered_result, direct_result)
        self.assertEqual(buffered_result, {0, 1})

    def test_remove(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)

        lsh.remove("a")
        self.assertTrue("a" not in lsh.keys)
        for table in lsh.hashtables:
            for H in table:
                self.assertGreater(len(table[H]), 0)
                self.assertTrue("a" not in table[H])

        self.assertRaises(ValueError, lsh.remove, "c")

    def test_pickle(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        lsh2 = pickle.loads(pickle.dumps(lsh))
        result = lsh2.query(m1)
        self.assertTrue("a" in result)
        result = lsh2.query(m2)
        self.assertTrue("b" in result)

    def test_insert_redis(self):
        with patch("redis.Redis", fake_redis):
            lsh = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )
            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")
            lsh.insert("a", m1)
            lsh.insert("b", m2)
            for t in lsh.hashtables:
                self.assertTrue(len(t) >= 1)
                items = []
                for H in t:
                    items.extend(t[H])
                self.assertTrue(pickle.dumps("a") in items)
                self.assertTrue(pickle.dumps("b") in items)
            self.assertTrue("a" in lsh)
            self.assertTrue("b" in lsh)
            for i, H in enumerate(lsh.keys[pickle.dumps("a")]):
                self.assertTrue(pickle.dumps("a") in lsh.hashtables[i][H])

            m3 = MinHash(18)
            self.assertRaises(ValueError, lsh.insert, "c", m3)

    def test_query_redis(self):
        with patch("redis.Redis", fake_redis):
            lsh = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )
            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")
            lsh.insert("a", m1)
            lsh.insert("b", m2)
            result = lsh.query(m1)
            self.assertTrue("a" in result)
            result = lsh.query(m2)
            self.assertTrue("b" in result)

            m3 = MinHash(18)
            self.assertRaises(ValueError, lsh.query, m3)

    def test_query_buffer_redis(self):
        with patch("redis.Redis", fake_redis):
            lsh = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )
            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")
            lsh.insert("a", m1)
            lsh.insert("b", m2)
            lsh.query(m1)
            lsh.add_to_query_buffer(m1)
            result = lsh.collect_query_buffer()
            self.assertTrue("a" in result)
            lsh.add_to_query_buffer(m2)
            result = lsh.collect_query_buffer()
            self.assertTrue("b" in result)

            m3 = MinHash(18)
            self.assertRaises(ValueError, lsh.add_to_query_buffer, m3)

    def test_insertion_session(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        data = [("a", m1), ("b", m2)]
        with lsh.insertion_session() as session:
            for key, minhash in data:
                session.insert(key, minhash)
        for t in lsh.hashtables:
            self.assertTrue(len(t) >= 1)
            items = []
            for H in t:
                items.extend(t[H])
            self.assertTrue("a" in items)
            self.assertTrue("b" in items)
        self.assertTrue("a" in lsh)
        self.assertTrue("b" in lsh)
        for i, H in enumerate(lsh.keys["a"]):
            self.assertTrue("a" in lsh.hashtables[i][H])

    def test_deletion_session(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        m3 = MinHash(16)
        m3.update(b"c")
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        lsh.insert("c", m3)

        keys_to_delete = ["a", "b"]
        with lsh.deletion_session() as session:
            for key in keys_to_delete:
                session.remove(key)

        # Verify deletions
        self.assertTrue("a" not in lsh.keys)
        self.assertTrue("b" not in lsh.keys)
        self.assertTrue("c" in lsh.keys)

        for table in lsh.hashtables:
            for H in table:
                self.assertTrue("a" not in table[H])
                self.assertTrue("b" not in table[H])

    def test_get_counts(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        counts = lsh.get_counts()
        self.assertEqual(len(counts), lsh.b)
        for table in counts:
            self.assertEqual(sum(table.values()), 2)

    def test_merge(self):
        lsh1 = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh1.insert("a", m1)
        lsh1.insert("b", m2)

        lsh2 = MinHashLSH(threshold=0.5, num_perm=16)
        m3 = MinHash(16)
        m3.update(b"c")
        m4 = MinHash(16)
        m4.update(b"d")
        lsh2.insert("c", m1)
        lsh2.insert("d", m2)

        lsh1.merge(lsh2)
        for t in lsh1.hashtables:
            self.assertTrue(len(t) >= 1)
            items = []
            for H in t:
                items.extend(t[H])
            self.assertTrue("c" in items)
            self.assertTrue("d" in items)
        self.assertTrue("a" in lsh1)
        self.assertTrue("b" in lsh1)
        self.assertTrue("c" in lsh1)
        self.assertTrue("d" in lsh1)
        for i, H in enumerate(lsh1.keys["c"]):
            self.assertTrue("c" in lsh1.hashtables[i][H])

        self.assertTrue(lsh1.merge, lsh2)
        self.assertRaises(ValueError, lsh1.merge, lsh2, check_overlap=True)

        m5 = MinHash(16)
        m5.update(b"e")
        lsh3 = MinHashLSH(threshold=0.5, num_perm=16)
        lsh3.insert("a", m5)

        self.assertRaises(ValueError, lsh1.merge, lsh3, check_overlap=True)

        lsh1.merge(lsh3)

        m6 = MinHash(16)
        m6.update(b"e")
        lsh4 = MinHashLSH(threshold=0.5, num_perm=16)
        lsh4.insert("a", m6)

        lsh1.merge(lsh4, check_overlap=False)

    def test_merge_rejects_different_band_hash_functions(self):
        minhash = MinHash(16)
        minhash.update(b"value")
        destination = MinHashLSH(threshold=0.5, num_perm=16, hashfunc=first_band_hash)
        source = MinHashLSH(threshold=0.5, num_perm=16, hashfunc=second_band_hash)
        source.insert("source", minhash)

        with self.assertRaisesRegex(ValueError, "different initialization parameters"):
            destination.merge(source)
        self.assertNotIn("source", destination)

    def test_merge_rejects_different_key_serialization(self):
        minhash = MinHash(16)
        minhash.update(b"value")
        destination = MinHashLSH(threshold=0.5, num_perm=16, prepickle=False)
        source = MinHashLSH(threshold=0.5, num_perm=16, prepickle=True)
        source.insert("source", minhash)

        with self.assertRaisesRegex(ValueError, "different initialization parameters"):
            destination.merge(source)
        self.assertNotIn("source", destination)

    def test_merge_redis(self):
        with patch("redis.Redis", fake_redis):
            lsh1 = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )
            lsh2 = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )

            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")
            lsh1.insert("a", m1)
            lsh1.insert("b", m2)

            m3 = MinHash(16)
            m3.update(b"c")
            m4 = MinHash(16)
            m4.update(b"d")
            lsh2.insert("c", m3)
            lsh2.insert("d", m4)

            lsh1.merge(lsh2)
            for t in lsh1.hashtables:
                self.assertTrue(len(t) >= 1)
                items = []
                for H in t:
                    items.extend(t[H])
                self.assertTrue(pickle.dumps("c") in items)
                self.assertTrue(pickle.dumps("d") in items)
            self.assertTrue("a" in lsh1)
            self.assertTrue("b" in lsh1)
            self.assertTrue("c" in lsh1)
            self.assertTrue("d" in lsh1)
            for i, H in enumerate(lsh1.keys[pickle.dumps("c")]):
                self.assertTrue(pickle.dumps("c") in lsh1.hashtables[i][H])

            self.assertTrue(lsh1.merge, lsh2)
            self.assertRaises(ValueError, lsh1.merge, lsh2, check_overlap=True)

            m5 = MinHash(16)
            m5.update(b"e")
            lsh3 = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )
            lsh3.insert("a", m5)

            self.assertRaises(ValueError, lsh1.merge, lsh3, check_overlap=True)

            m6 = MinHash(16)
            m6.update(b"e")
            lsh4 = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )
            lsh4.insert("a", m6)

            lsh1.merge(lsh4, check_overlap=False)

    def test_redis_deletion_session(self):
        with patch("redis.Redis", fake_redis):
            lsh = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )

            # The library's RedisBuffer is not completely compatible with mockredis.
            # In order to account for this, we:
            # 1. Replace the buffer with mockredis's own simple pipeline, which can execute.
            # 2. Patch __init__ to prevent the buffer flush from resetting the storage object
            #    and wiping mock data.
            storage_objects = [lsh.keys, *lsh.hashtables]
            for storage in storage_objects:
                storage._buffer = storage._redis.pipeline()

            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")
            m3 = MinHash(16)
            m3.update(b"c")
            lsh.insert("a", m1)
            lsh.insert("b", m2)
            lsh.insert("c", m3)

            keys_to_delete = ["a", "b"]

            with (
                patch.object(type(lsh.keys), "__init__", lambda self, config, name: None),
                lsh.deletion_session() as session,
            ):
                for key in keys_to_delete:
                    session.remove(key)

            # Verify deletions
            self.assertTrue("a" not in lsh)
            self.assertTrue("b" not in lsh)
            self.assertTrue("c" in lsh)

            # Verify underlying storage
            for table in lsh.hashtables:
                for H in table:
                    items = table[H]
                    self.assertTrue("a".encode("utf8") not in items)
                    self.assertTrue("b".encode("utf8") not in items)


class TestWeightedMinHashLSH(unittest.TestCase):
    def test_init(self):
        lsh = MinHashLSH(threshold=0.8)
        self.assertTrue(lsh.is_empty())
        b1, r1 = lsh.b, lsh.r
        lsh = MinHashLSH(threshold=0.8, weights=(0.2, 0.8))
        b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)

    def test__H(self):
        """Check _H output consistent bytes length given
        the same concatenated hash value size.
        """
        mg = WeightedMinHashGenerator(100, sample_size=128)
        for _l in range(2, mg.sample_size + 1, 16):
            m = mg.minhash(np.random.randint(1, 99999999, 100))
            lsh = MinHashLSH(num_perm=128)
            lsh.insert("m", m)
            sizes = [len(H) for ht in lsh.hashtables for H in ht]
            self.assertTrue(all(sizes[0] == s for s in sizes))

    def test_insert(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=4)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        for t in lsh.hashtables:
            self.assertTrue(len(t) >= 1)
            items = []
            for H in t:
                items.extend(t[H])
            self.assertTrue("a" in items)
            self.assertTrue("b" in items)
        self.assertTrue("a" in lsh)
        self.assertTrue("b" in lsh)
        for i, H in enumerate(lsh.keys["a"]):
            self.assertTrue("a" in lsh.hashtables[i][H])

        mg = WeightedMinHashGenerator(10, 5)
        m3 = mg.minhash(np.random.uniform(1, 10, 10))
        self.assertRaises(ValueError, lsh.insert, "c", m3)

    def test_query(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=4)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        result = lsh.query(m1)
        self.assertTrue("a" in result)
        result = lsh.query(m2)
        self.assertTrue("b" in result)

        mg = WeightedMinHashGenerator(10, 5)
        m3 = mg.minhash(np.random.uniform(1, 10, 10))
        self.assertRaises(ValueError, lsh.query, m3)

    def test_remove(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=4)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        lsh.insert("a", m1)
        lsh.insert("b", m2)

        lsh.remove("a")
        self.assertTrue("a" not in lsh.keys)
        for table in lsh.hashtables:
            for H in table:
                self.assertGreater(len(table[H]), 0)
                self.assertTrue("a" not in table[H])

        self.assertRaises(ValueError, lsh.remove, "c")

    def test_pickle(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=4)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        lsh2 = pickle.loads(pickle.dumps(lsh))

        result = lsh2.query(m1)
        self.assertTrue("a" in result)
        result = lsh2.query(m2)
        self.assertTrue("b" in result)


class InjectedStorageError(RuntimeError):
    """Raised by ProbedStorage when the configured Nth operation is reached."""


class StorageProbe:
    """Records the order of mutating storage operations and injects one failure.

    The probe fails the Nth (1-based) call of `fail_method` made through a
    wrapper tagged `fail_tag`, letting a test target e.g. the 3rd bucket
    write across all band tables.
    """

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


class ProbedStorage:
    """Wraps a storage instance, logging mutating calls to a StorageProbe."""

    def __init__(self, inner, probe, tag):
        self._inner = inner
        self._probe = probe
        self._tag = tag

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __contains__(self, key):
        return key in self._inner

    def __getitem__(self, key):
        return self._inner[key]

    def __iter__(self):
        return iter(self._inner)

    def __len__(self):
        return len(self._inner)

    def insert(self, *args, **kwargs):
        self._probe.record(self._tag, "insert")
        return self._inner.insert(*args, **kwargs)

    def remove(self, *args, **kwargs):
        self._probe.record(self._tag, "remove")
        return self._inner.remove(*args, **kwargs)

    def remove_val(self, *args, **kwargs):
        self._probe.record(self._tag, "remove_val")
        return self._inner.remove_val(*args, **kwargs)


class TestMinHashLSHFailureStates(unittest.TestCase):
    """Pin the insert/remove state machine of MinHashLSH against a fake
    storage that fails on the Nth bucket write or delete.

    The core invariant under test: after any internal failure, a key must
    not be left in only one direction of the index (key -> Hs record without
    bucket entries, or bucket entries without the key record). The in-memory
    dict backend has no transaction support; the guarantees checked here come
    from the LSH layer itself and therefore apply to every backend.
    """

    def _make_lsh(self, num_perm=16):
        lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)
        probe = StorageProbe()
        lsh.hashtables = [ProbedStorage(t, probe, "bucket") for t in lsh.hashtables]
        lsh.keys = ProbedStorage(lsh.keys, probe, "keys")
        return lsh, probe

    @staticmethod
    def _minhash(content, num_perm=16):
        m = MinHash(num_perm)
        m.update(content)
        return m

    def _assert_present_both_directions(self, lsh, key):
        stored_Hs = list(lsh.keys.get(key))
        self.assertEqual(len(stored_Hs), lsh.b, "key -> Hs record must hold one H per band")
        for H, table in zip(stored_Hs, lsh.hashtables):
            self.assertIn(key, list(table.get(H)), "bucket entry missing for band")

    def _assert_absent_both_directions(self, lsh, key):
        self.assertEqual(list(lsh.keys.get(key)), [], "key -> Hs record must be gone")
        for table in lsh.hashtables:
            for H in table:
                self.assertNotIn(key, list(table.get(H)), "bucket entry left behind")

    def test_insert_writes_key_record_before_bucket_entries(self):
        lsh, probe = self._make_lsh()
        m = self._minhash(b"a")
        lsh.insert("a", m)
        self.assertEqual(probe.ops[0], ("keys", "insert"))
        self.assertEqual(probe.ops[1:], [("bucket", "insert")] * lsh.b)
        self._assert_present_both_directions(lsh, "a")
        self.assertIn("a", lsh.query(m))

    def test_insertion_session_unclosed_dict_backend_applies_writes_immediately(self):
        # The dict backend ignores `buffer`, so writes are visible before
        # close(); this documents the in-memory state machine only. Buffered
        # backends (e.g. Redis pipelines) flush at session close instead.
        lsh, _ = self._make_lsh()
        m = self._minhash(b"a")
        session = lsh.insertion_session()
        session.insert("a", m)
        self.assertIn("a", lsh.query(m))
        self._assert_present_both_directions(lsh, "a")
        session.close()
        session.close()  # close is idempotent
        self._assert_present_both_directions(lsh, "a")

    def test_insert_duplicate_key_raises_before_touching_storage(self):
        lsh, probe = self._make_lsh()
        m = self._minhash(b"a")
        lsh.insert("a", m)
        ops_before = list(probe.ops)
        self.assertRaises(ValueError, lsh.insert, "a", m)
        self.assertEqual(probe.ops, ops_before, "duplicate check must reject before any storage write")
        self._assert_present_both_directions(lsh, "a")
        self.assertIn("a", lsh.query(m))

    def test_insert_failure_at_nth_bucket_write_leaves_no_one_directional_residue(self):
        for fail_at in (1, 3, 5):  # first band, middle band, last band (b=5 for num_perm=16)
            lsh, probe = self._make_lsh()
            self.assertEqual(lsh.b, 5)
            m = self._minhash(b"x")
            lsh.insert("ok", self._minhash(b"ok"))
            probe.arm("bucket", "insert", fail_at)
            with self.assertRaises(InjectedStorageError, msg=f"failure at bucket write #{fail_at}"):
                lsh.insert("x", m)
            # The failed key must not survive in only one direction.
            self._assert_absent_both_directions(lsh, "x")
            self.assertNotIn("x", lsh.query(m))
            # Compensation must appear in the recorded op order.
            compensated = probe.ops.count(("bucket", "remove_val"))
            self.assertEqual(compensated, fail_at - 1, "every landed bucket write must be compensated")
            self.assertEqual(probe.ops[-1], ("keys", "remove"))
            # The pre-existing key is untouched, and the index accepts the
            # same key again after the blocked attempt.
            self._assert_present_both_directions(lsh, "ok")
            lsh.insert("x", m)
            self._assert_present_both_directions(lsh, "x")
            self.assertIn("x", lsh.query(m))

    def test_insert_failure_rollback_preserves_colliding_key(self):
        lsh, probe = self._make_lsh()
        m = self._minhash(b"shared")
        lsh.insert("good", m)  # same minhash -> shares every bucket with "bad"
        probe.arm("bucket", "insert", 3)
        with self.assertRaises(InjectedStorageError):
            lsh.insert("bad", m)
        # Rolling back the failed key must not evict the colliding good key.
        self._assert_present_both_directions(lsh, "good")
        self.assertIn("good", lsh.query(m))
        self._assert_absent_both_directions(lsh, "bad")

    def test_remove_failure_at_nth_bucket_delete_restores_both_directions(self):
        for fail_at in (1, 3, 5):
            lsh, probe = self._make_lsh()
            m = self._minhash(b"x")
            lsh.insert("x", m)
            probe.arm("bucket", "remove_val", fail_at)
            with self.assertRaises(InjectedStorageError, msg=f"failure at bucket delete #{fail_at}"):
                lsh.remove("x")
            # The blocked remove must leave the key fully indexed, not
            # stranded in only one direction.
            self._assert_present_both_directions(lsh, "x")
            self.assertIn("x", lsh.query(m))
            # A retry after the blocked remove completes the cleanup.
            lsh.remove("x")
            self._assert_absent_both_directions(lsh, "x")
            self.assertNotIn("x", lsh.query(m))
            self.assertTrue(lsh.is_empty())

    def test_remove_failure_at_key_record_delete_restores_buckets(self):
        lsh, probe = self._make_lsh()
        m = self._minhash(b"x")
        lsh.insert("x", m)
        probe.arm("keys", "remove", 1)
        with self.assertRaises(InjectedStorageError):
            lsh.remove("x")
        self._assert_present_both_directions(lsh, "x")
        self.assertIn("x", lsh.query(m))
        lsh.remove("x")
        self._assert_absent_both_directions(lsh, "x")

    def test_remove_missing_key_raises_without_touching_storage(self):
        lsh, probe = self._make_lsh()
        lsh.insert("a", self._minhash(b"a"))
        ops_before = list(probe.ops)
        self.assertRaises(ValueError, lsh.remove, "missing")
        self.assertEqual(probe.ops, ops_before, "missing-key check must reject before any storage write")
        self._assert_present_both_directions(lsh, "a")

    def test_deletion_session_failure_then_successful_cleanup(self):
        lsh, probe = self._make_lsh()
        m = self._minhash(b"x")
        lsh.insert("x", m)
        lsh.insert("y", self._minhash(b"y"))
        probe.arm("bucket", "remove_val", 2)
        session = lsh.deletion_session()
        with self.assertRaises(InjectedStorageError):
            session.remove("x")
        session.close()
        self._assert_present_both_directions(lsh, "x")
        with lsh.deletion_session() as session:
            session.remove("x")
        self._assert_absent_both_directions(lsh, "x")
        self._assert_present_both_directions(lsh, "y")


if __name__ == "__main__":
    unittest.main()
