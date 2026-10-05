import importlib
import os

import fsspec
import pytest
from fsspec.implementations.memory import MemoryFileSystem

import intake
from intake.readers import datatypes, readers

pd = pytest.importorskip("pandas")
csv_bytes = b"value\n1\n2\n"


@pytest.fixture
def fixture_https(monkeypatch):
    class FixtureHTTPS(MemoryFileSystem):
        protocol = "https"
        store = {}
        pseudo_dirs = [""]

        @classmethod
        def _strip_protocol(cls, path):
            return MemoryFileSystem._strip_protocol(str(path).removeprefix("https://"))

    registry = importlib.import_module("fsspec.registry")
    monkeypatch.setitem(registry._registry, "https", FixtureHTTPS)
    fs = fsspec.filesystem("https")
    fs.pipe("https://fixture.test/table.csv", csv_bytes)
    yield fs
    FixtureHTTPS.clear_instance_cache()


@pytest.mark.parametrize("wrapper", ["simplecache", "simplecache://", "filecache"])
@pytest.mark.parametrize("head", [True, False, csv_bytes])
def test_cached_http_csv_reader(wrapper, head, fixture_https, tmp_path):
    url = f"{wrapper}::https://fixture.test/table.csv"
    options = {wrapper.removesuffix("://"): {"cache_storage": str(tmp_path / "cache")}}
    with fsspec.open(url, **options) as f:
        assert f.read() == csv_bytes

    datatype = intake.recommend(url, head=head, storage_options=options)[0]
    assert datatype is datatypes.CSV
    reader = datatype(url, storage_options=options).to_reader(outtype="pandas:DataFrame")
    assert isinstance(reader, readers.PandasCSV)
    assert reader.data.url == url
    assert reader.data.storage_options is options
    assert reader.read()["value"].tolist() == [1, 2]


@pytest.mark.skipif(os.name == "nt", reason="Colons are not valid in Windows filenames")
@pytest.mark.parametrize("head", [True, False, csv_bytes])
def test_local_csv_with_double_colon(tmp_path, head):
    filename = tmp_path / "observations::2026.csv"
    filename.write_bytes(csv_bytes)
    url = str(filename)
    assert pd.read_csv(url)["value"].tolist() == [1, 2]

    datatype = intake.recommend(url, head=head)[0]
    assert datatype is datatypes.CSV
    reader = datatype(url).to_reader(outtype="pandas:DataFrame")
    assert reader.data.url == url
    assert reader.read()["value"].tolist() == [1, 2]


@pytest.mark.parametrize("wrapper", ["", "simplecache::"])
@pytest.mark.parametrize("address", ["[::1]", "[2001:db8::1]"])
def test_ipv6_csv_filename(address, wrapper, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Filename-only inference must not access a filesystem")

    monkeypatch.setattr(fsspec.core, "url_to_fs", forbidden)
    url = f"{wrapper}https://{address}/table.csv"
    recommended = intake.recommend(url, head=False)
    assert datatypes.CSV in recommended
    if wrapper:
        assert recommended[0] is datatypes.CSV
