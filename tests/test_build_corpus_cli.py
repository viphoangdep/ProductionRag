from unittest.mock import patch, MagicMock
from scripts.build_corpus import main


@patch("scripts.build_corpus.CorpusService")
def test_cli_invokes_service_build_sync(mock_service_cls):
    mock_inst = MagicMock()
    mock_inst.build_corpus_sync.return_value = {"doc1": {}}
    mock_service_cls.return_value = mock_inst

    code = main(["--docs", "docs", "--version", "v2", "--skip-s3"])
    assert code == 0
    assert mock_inst.build_corpus_sync.called
    kwargs = mock_inst.build_corpus_sync.call_args.kwargs
    assert kwargs["version"] == "v2"
    assert kwargs["skip_s3"] is True

