from src.utils.backup import PeriodicBackup, local_copy_backup


def test_local_copy_backup_copies_files(tmp_path):
    source = tmp_path / "results" / "checkpoints"
    source.mkdir(parents=True)
    (source / "registry.json").write_text('{"a": 1}')

    dest = tmp_path / "backup"
    local_copy_backup([source], dest)

    copied = dest / "checkpoints" / "registry.json"
    assert copied.exists()
    assert copied.read_text() == '{"a": 1}'


def test_local_copy_backup_overwrites_on_repeat_calls(tmp_path):
    source = tmp_path / "results" / "checkpoints"
    source.mkdir(parents=True)
    (source / "registry.json").write_text('{"a": 1}')
    dest = tmp_path / "backup"

    local_copy_backup([source], dest)
    (source / "registry.json").write_text('{"a": 2}')
    local_copy_backup([source], dest)  # second call must not raise on existing dest

    assert (dest / "checkpoints" / "registry.json").read_text() == '{"a": 2}'


def test_local_copy_backup_missing_source_does_not_raise(tmp_path):
    dest = tmp_path / "backup"
    local_copy_backup([tmp_path / "does_not_exist"], dest)  # should log + skip, not raise


def test_local_copy_backup_multiple_sources(tmp_path):
    checkpoints = tmp_path / "results" / "checkpoints"
    pools = tmp_path / "results" / "pools"
    checkpoints.mkdir(parents=True)
    pools.mkdir(parents=True)
    (checkpoints / "a.json").write_text("x")
    (pools / "b.jsonl").write_text("y")

    dest = tmp_path / "backup"
    local_copy_backup([checkpoints, pools], dest)

    assert (dest / "checkpoints" / "a.json").exists()
    assert (dest / "pools" / "b.jsonl").exists()


def test_periodic_backup_triggers_at_correct_interval(tmp_path):
    calls = []

    class FakeBackup(PeriodicBackup):
        def maybe_backup(self):
            triggered = super().maybe_backup()
            if triggered:
                calls.append(self._counter)
            return triggered

    pb = FakeBackup(every_n=3, local_dest=tmp_path, source_dirs=[])
    results = [pb.maybe_backup() for _ in range(9)]

    assert results == [False, False, True, False, False, True, False, False, True]
    assert calls == [3, 6, 9]


def test_periodic_backup_disabled_with_zero_never_triggers():
    pb = PeriodicBackup(every_n=0, local_dest=None, source_dirs=[])
    results = [pb.maybe_backup() for _ in range(10)]
    assert not any(results)


def test_periodic_backup_actually_calls_local_copy(tmp_path):
    source = tmp_path / "results" / "checkpoints"
    source.mkdir(parents=True)
    (source / "registry.json").write_text("data")
    dest = tmp_path / "backup"

    pb = PeriodicBackup(every_n=1, local_dest=dest, source_dirs=[source])
    pb.maybe_backup()

    assert (dest / "checkpoints" / "registry.json").exists()
