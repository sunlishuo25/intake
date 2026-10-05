import os
import zipfile

import pytest

import intake
from intake.readers import datatypes, readers, entry

here = os.path.dirname(__file__)
testdir = os.path.abspath(os.path.join(here, "..", "..", "catalog/tests"))


def test1():
    data = datatypes.CSV(url=f"{testdir}/entry1_1.csv")
    reader = readers.PandasCSV(data)
    assert reader.doc()
    out = reader.read()
    assert list(out.columns) == ["name", "score", "rank"]


def test_recommend_filetype():
    assert datatypes.Parquet in datatypes.recommend(url="myfile.parq")
    assert datatypes.Parquet in datatypes.recommend(head=b"PAR1")
    assert all(
        p in datatypes.recommend(mime="text/yaml", head=False)
        for p in {datatypes.YAMLFile, datatypes.CatalogFile}
    )


@pytest.mark.parametrize(
    "url, expected, unexpected",
    [
        ("zip://table.csv::file:///archive.zip", datatypes.CSV, datatypes.Parquet),
        ("zip://table.csv::https://example.com/archive.parquet", datatypes.CSV, datatypes.Parquet),
        ("zip://table.parquet::https://example.com/archive.csv", datatypes.Parquet, datatypes.CSV),
        (
            "simplecache::zip://table.csv::https://example.com/archive.zip",
            datatypes.CSV,
            datatypes.Parquet,
        ),
        ("simplecache::https://example.com/table.csv", datatypes.CSV, datatypes.Parquet),
        ("simplecache://::https://example.com/table.csv", datatypes.CSV, datatypes.Parquet),
        ("customcache::https://example.com/table.csv", datatypes.CSV, datatypes.Parquet),
    ],
)
def test_recommend_chained_url(url, expected, unexpected):
    recommended = datatypes.recommend(url=url, head=False)
    assert expected in recommended
    assert unexpected not in recommended


def test_recommend_chained_url_read(tmp_path):
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("table.csv", "value\n1\n2\n")
    url = f"zip://table.csv::{archive}"

    datatype = intake.recommend(url)[0]
    reader = datatype(url).to_reader(outtype="pandas:DataFrame")
    assert isinstance(reader, readers.PandasCSV)
    assert reader.data.url == url
    assert reader.read()["value"].tolist() == [1, 2]


def test_recommend_reader():
    pp = datatypes.Parquet("")
    rec = readers.recommend(pp)
    assert all(p not in rec["importable"] for p in rec["not_importable"])
    assert all(p not in rec["not_importable"] for p in rec["importable"])
    assert all(
        p in rec["importable"] + rec["not_importable"]
        for p in {readers.PandasParquet, readers.AwkwardParquet, readers.DaskParquet}
    )
    pp = datatypes.CSV("")
    assert readers.PandasCSV in readers.recommend(pp)["importable"]


def test_data_metadata():
    cat = entry.Catalog()
    cat["d"] = datatypes.BaseData(metadata={"oi": "io"})
    cat.get_entity("d").metadata.update(blag=0)
    out = cat["d"]
    assert out.metadata == {"oi": "io", "blag": 0}
