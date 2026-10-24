"""The package was called posbench up to rules v1.2: the former command name runs the same command line after a
one-line notice, and the former name of the genome-registry variable is still read."""
import pytest

from seqleak.audit import genome as G
from seqleak.cli import main_posbench


def test_posbench_command(capsys):
    with pytest.raises(SystemExit) as e:
        main_posbench(['--help'])
    assert e.value.code == 0
    out = capsys.readouterr()
    assert out.err.startswith('posbench has been renamed seqleak') and 'audit' in out.out


def test_registry_variable(tmp_path, monkeypatch):
    (tmp_path / 'r.fa').write_text('>c\nACGT\n')
    (tmp_path / 'reg.tsv').write_text(f'toy\t{tmp_path / "r.fa"}\n')
    monkeypatch.delenv('SEQLEAK_GENOME_REGISTRY', raising=False)
    monkeypatch.setenv('POSBENCH_GENOME_REGISTRY', str(tmp_path / 'reg.tsv'))
    assert G.resolve(['toy']) == [str(tmp_path / 'r.fa')]
    monkeypatch.setenv('SEQLEAK_GENOME_REGISTRY', str(tmp_path / 'reg.tsv'))
    monkeypatch.setenv('POSBENCH_GENOME_REGISTRY', str(tmp_path / 'missing.tsv'))
    assert G.resolve(['toy']) == [str(tmp_path / 'r.fa')]
