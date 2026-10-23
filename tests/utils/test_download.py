"""Unit tests for download utility functions."""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import filelock
import pytest
import requests

from gwmock.utils.download import determine_dest_path, download_file, download_file_with_lock, handle_existing_file
from gwmock.utils.errors import DeterministicRefusalError, TransientDownloadError


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_response():
    """Mock requests.Response object."""
    response = MagicMock(spec=requests.Response)
    response.status_code = 200
    response.headers = {}
    # Return an iterator that yields data chunks
    response.iter_content = MagicMock(return_value=iter([b"test data"]))
    response.raise_for_status.return_value = None
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=None)
    return response


def test_download_file_basic(temp_dir, mocker, mock_response):
    """Test basic file download."""
    url = "https://example.com/file.txt"
    mock_response.headers = {"Content-Type": "application/octet-stream"}
    mocker.patch("gwmock.utils.download.requests.get", return_value=mock_response)

    dest_path = download_file(url, outdir=temp_dir)

    assert dest_path.exists()
    assert dest_path.name == "file.txt"
    with dest_path.open("rb") as f:
        assert f.read() == b"test data"


def test_download_file_with_dest_path(temp_dir, mocker, mock_response):
    """Test download with specified dest_path."""
    url = "https://example.com/file.txt"
    mocker.patch("gwmock.utils.download.requests.get", return_value=mock_response)

    dest_path = download_file(url, dest_path="custom.txt", outdir=temp_dir)

    assert dest_path.exists()
    assert dest_path.name == "custom.txt"


def test_download_file_hashed_url(temp_dir, mocker, mock_response):
    """Test download with hashed URL filename."""
    url = "https://example.com/file.txt"
    mocker.patch("gwmock.utils.download.requests.get", return_value=mock_response)

    dest_path = download_file(url, dest_path_from_hashed_url=True, outdir=temp_dir)

    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
    assert dest_path.name == f"{url_hash}.txt"


def test_download_file_allow_existing(temp_dir, mocker, mock_response):
    """Test skipping download if file exists and allow_existing=True."""
    url = "https://example.com/file.txt"
    existing_file = temp_dir / "file.txt"
    existing_file.write_text("existing")

    dest_path = download_file(url, outdir=temp_dir, allow_existing=True)

    assert dest_path == existing_file
    # Ensure requests.get was not called
    mocker.patch("requests.get", return_value=mock_response)
    # Since file exists, no download should happen


def test_download_file_overwrite_false_existing(temp_dir, mocker, mock_response):
    """Test raising error if file exists and overwrite=False, allow_existing=False."""
    url = "https://example.com/file.txt"
    existing_file = temp_dir / "file.txt"
    existing_file.write_text("existing")

    with pytest.raises(FileExistsError):
        download_file(url, outdir=temp_dir, overwrite=False, allow_existing=False)


def test_download_file_infer_extension(temp_dir, mocker):
    """Test inferring file extension from Content-Type."""
    url = "https://example.com/file"
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/plain"}
    mock_response.iter_content = MagicMock(return_value=iter([b"test data"]))
    mock_response.raise_for_status.return_value = None
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)
    mocker.patch("gwmock.utils.download.requests.get", return_value=mock_response)

    dest_path = download_file(url, outdir=temp_dir)

    assert dest_path.suffix == ".txt"


def test_download_file_request_exception(temp_dir, mocker):
    """Test handling of request exceptions."""
    url = "https://example.com/file.txt"
    mocker.patch("gwmock.utils.download.requests.get", side_effect=requests.RequestException("Network error"))

    with pytest.raises(TransientDownloadError, match="Failed to download file"):
        download_file(url, outdir=temp_dir)


def test_download_file_lock_timeout(temp_dir, mocker, mock_response):
    """Test handling of lock timeout."""
    url = "https://example.com/file.txt"
    mocker.patch("gwmock.utils.download.requests.get", return_value=mock_response)
    mocker.patch("gwmock.utils.download.filelock.FileLock", side_effect=filelock.Timeout("Lock timeout"))

    with pytest.raises(TransientDownloadError, match="Timeout waiting for download lock"):
        download_file(url, outdir=temp_dir, timeout=1)


def test_download_file_no_extension_fallback(temp_dir, mocker):
    """Test fallback to .bin when no extension can be inferred."""
    url = "https://example.com/file"
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/octet-stream"}
    mock_response.iter_content = MagicMock(return_value=iter([b"test data"]))
    mock_response.raise_for_status.return_value = None
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)
    mocker.patch("gwmock.utils.download.requests.get", return_value=mock_response)

    dest_path = download_file(url, outdir=temp_dir)

    assert dest_path.suffix == ".bin"


def test_determine_dest_path(temp_dir):
    """Test determining destination path from URL."""
    url = "https://example.com/file.txt"
    dest_path = determine_dest_path(url, outdir=temp_dir)
    assert dest_path == temp_dir / "file.txt"


def test_determine_dest_path_hashed(temp_dir):
    """Test determining destination path with hashed URL."""
    url = "https://example.com/file.txt"
    dest_path = determine_dest_path(url, outdir=temp_dir, dest_path_from_hashed_url=True)
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
    assert dest_path == temp_dir / f"{url_hash}.txt"


def test_determine_dest_path_with_dest_path(temp_dir):
    """Test determining destination path with provided dest_path."""
    url = "https://example.com/file.txt"
    dest_path = determine_dest_path(url, dest_path="custom.txt", outdir=temp_dir)
    assert dest_path == temp_dir / "custom.txt"


def test_handle_existing_file_allow_existing(temp_dir):
    """Test handle_existing_file with allow_existing=True."""
    dest_path = temp_dir / "file.txt"
    dest_path.write_text("existing")
    result = handle_existing_file(dest_path, overwrite=False, allow_existing=True)
    assert result == dest_path


def test_handle_existing_file_overwrite_false(temp_dir):
    """Test handle_existing_file with overwrite=False, allow_existing=False."""
    dest_path = temp_dir / "file.txt"
    dest_path.write_text("existing")
    with pytest.raises(FileExistsError):
        handle_existing_file(dest_path, overwrite=False, allow_existing=False)


def test_handle_existing_file_no_file(temp_dir):
    """Test handle_existing_file when file does not exist."""
    dest_path = temp_dir / "file.txt"
    result = handle_existing_file(dest_path, overwrite=False, allow_existing=True)
    assert result is None


def test_download_file_with_lock(temp_dir, mocker, mock_response):
    """Test download_file_with_lock function."""
    url = "https://example.com/file.txt"
    dest_path = temp_dir / "file.txt"
    lock_path = dest_path.with_suffix(dest_path.suffix + ".lock")
    mock_response.headers = {"Content-Type": "application/octet-stream"}
    mocker.patch("gwmock.utils.download.requests.get", return_value=mock_response)

    result = download_file_with_lock(url, dest_path, lock_path, timeout=300)

    assert result == dest_path
    assert dest_path.exists()
    with dest_path.open("rb") as f:
        assert f.read() == b"test data"


class TestWhatTheDownloadIsActuallyToldToDo:
    """The arguments the helpers pass on, rather than only the file that comes out.

    ``requests.get`` and ``FileLock`` are mocked in every test above, so a mutation to what they
    are *given* is invisible to an assertion about the downloaded bytes -- and two of those
    arguments decide whether a stalled server or a stale lock blocks the caller forever.
    """

    @staticmethod
    def _capture_get(mocker, mock_response):
        mock_response.headers = {"Content-Type": "application/octet-stream"}
        return mocker.patch("gwmock.utils.download.requests.get", return_value=mock_response)

    def test_the_url_reaches_requests(self, temp_dir, mocker, mock_response):
        get = self._capture_get(mocker, mock_response)
        download_file("https://example.com/file.txt", outdir=temp_dir)
        assert get.call_args.args[0] == "https://example.com/file.txt"

    def test_the_body_is_streamed(self, temp_dir, mocker, mock_response):
        """``stream=True`` keeps a multi-hundred-megabyte catalogue out of memory; the write loop
        below it iterates the response in chunks and only works on a streamed response."""
        get = self._capture_get(mocker, mock_response)
        download_file("https://example.com/file.txt", outdir=temp_dir)
        assert get.call_args.kwargs["stream"] is True

    def test_the_request_carries_the_timeout(self, temp_dir, mocker, mock_response):
        """Without it a server that accepts the connection and then says nothing hangs the run."""
        get = self._capture_get(mocker, mock_response)
        download_file("https://example.com/file.txt", outdir=temp_dir, timeout=17)
        assert get.call_args.kwargs["timeout"] == 17

    def test_the_default_timeout_is_five_minutes(self, temp_dir, mocker, mock_response):
        get = self._capture_get(mocker, mock_response)
        download_file("https://example.com/file.txt", outdir=temp_dir)
        assert get.call_args.kwargs["timeout"] == 300

    def test_the_lock_is_taken_with_a_timeout(self, temp_dir, mocker, mock_response):
        """A ``FileLock`` built without one blocks forever by default, so a lock left behind by a
        killed process would stall every later run instead of failing with the message below."""
        self._capture_get(mocker, mock_response)
        lock = mocker.patch("gwmock.utils.download.filelock.FileLock")
        download_file("https://example.com/file.txt", outdir=temp_dir, timeout=42)
        assert lock.call_args.kwargs["timeout"] == 42

    def test_the_lock_sits_beside_the_file_it_guards(self, temp_dir, mocker, mock_response):
        """Two downloads of the same URL must contend on the same path; the suffix is appended to
        the whole name rather than replacing it, so ``a.txt`` and ``a.csv`` do not share a lock."""
        self._capture_get(mocker, mock_response)
        lock = mocker.patch("gwmock.utils.download.filelock.FileLock")
        download_file("https://example.com/file.txt", outdir=temp_dir)
        assert Path(lock.call_args.args[0]) == temp_dir / "file.txt.lock"

    def test_a_lock_timeout_names_the_url(self, temp_dir, mocker, mock_response):
        self._capture_get(mocker, mock_response)
        mocker.patch("gwmock.utils.download.filelock.FileLock", side_effect=filelock.Timeout("held elsewhere"))
        with pytest.raises(TransientDownloadError, match=re.escape("https://example.com/file.txt")):
            download_file("https://example.com/file.txt", outdir=temp_dir)


class TestExistingFilesAndDefaults:
    def test_an_existing_file_is_kept_by_default(self, temp_dir):
        """The default is permissive on purpose: a resumed run finds its inputs already there and
        must carry on rather than refuse. Flipping the default turns every resume into an error."""
        dest_path = temp_dir / "file.txt"
        dest_path.write_text("existing")
        assert handle_existing_file(dest_path, overwrite=False) == dest_path

    def test_download_file_keeps_an_existing_file_by_default(self, temp_dir, mocker, mock_response):
        get = mocker.patch("gwmock.utils.download.requests.get", return_value=mock_response)
        dest_path = temp_dir / "file.txt"
        dest_path.write_text("existing")

        result = download_file("https://example.com/file.txt", outdir=temp_dir)

        assert result == dest_path
        assert dest_path.read_text() == "existing"
        get.assert_not_called()

    def test_overwrite_downloads_over_an_existing_file(self, temp_dir, mocker, mock_response):
        """``overwrite`` has to reach the existence check as a real boolean: anything truthy-but-
        wrong there (``None`` for instance) makes ``not overwrite`` true again and the request is
        never made."""
        mock_response.headers = {"Content-Type": "application/octet-stream"}
        get = mocker.patch("gwmock.utils.download.requests.get", return_value=mock_response)
        dest_path = temp_dir / "file.txt"
        dest_path.write_text("existing")

        result = download_file("https://example.com/file.txt", outdir=temp_dir, overwrite=True)

        assert result == dest_path
        assert dest_path.read_text() == "test data"
        get.assert_called_once()

    def test_a_refusal_names_the_file_and_the_flag(self, temp_dir):
        dest_path = temp_dir / "file.txt"
        dest_path.write_text("existing")
        with pytest.raises(FileExistsError, match=r"file\.txt already exists and overwrite is set to False"):
            handle_existing_file(dest_path, overwrite=False, allow_existing=False)


class TestTheSuffixWhenTheUrlHasNone:
    def test_a_missing_content_type_falls_back_to_bin(self, temp_dir, mocker, mock_response):
        """No header at all: the fallback has to be a suffix, because the caller writes to the path
        it gets back and an empty suffix would collide with the directory-style URL name."""
        mock_response.headers = {}
        mocker.patch("gwmock.utils.download.requests.get", return_value=mock_response)
        assert download_file("https://example.com/data", outdir=temp_dir).suffix == ".bin"

    def test_an_unrecognised_content_type_falls_back_to_bin(self, temp_dir, mocker, mock_response):
        mock_response.headers = {"Content-Type": "application/x-not-a-real-type"}
        mocker.patch("gwmock.utils.download.requests.get", return_value=mock_response)
        assert download_file("https://example.com/data", outdir=temp_dir).suffix == ".bin"

    def test_the_content_type_decides_the_suffix(self, temp_dir, mocker, mock_response):
        mock_response.headers = {"Content-Type": "text/plain"}
        mocker.patch("gwmock.utils.download.requests.get", return_value=mock_response)
        assert download_file("https://example.com/data", outdir=temp_dir).suffix == ".txt"

    def test_parameters_after_the_media_type_are_ignored(self, temp_dir, mocker, mock_response):
        """A real server sends ``text/plain; charset=utf-8``, which is not a media type on its own."""
        mock_response.headers = {"Content-Type": "text/plain; charset=utf-8"}
        mocker.patch("gwmock.utils.download.requests.get", return_value=mock_response)
        assert download_file("https://example.com/data", outdir=temp_dir).suffix == ".txt"


class TestAnInterruptedOverwrite:
    """A failed overwrite must not destroy the file it was replacing.

    ``overwrite=True`` is the only path that writes over an existing file, so it is the only one
    where a stream that dies part-way can lose data. The download is staged next to the target and
    moved into place once every chunk has arrived.
    """

    @staticmethod
    def _failing_response(mocker, chunks):
        response = MagicMock(spec=requests.Response)
        response.status_code = 200
        response.headers = {"Content-Type": "application/octet-stream"}
        response.raise_for_status.return_value = None
        response.iter_content = MagicMock(return_value=iter(chunks))
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=None)
        return mocker.patch("gwmock.utils.download.requests.get", return_value=response)

    @staticmethod
    def _files_ignoring_the_lock(directory):
        """The lock file is the download's own bookkeeping and outlives the transfer."""
        return sorted(p.name for p in directory.iterdir() if p.suffix != ".lock")

    @staticmethod
    def _dying_chunks():
        yield b"partial"
        raise requests.ConnectionError("connection reset mid-transfer")

    def test_the_previous_file_survives_a_stream_that_dies(self, temp_dir, mocker):
        dest_path = temp_dir / "file.txt"
        dest_path.write_text("the good copy")
        self._failing_response(mocker, self._dying_chunks())

        with pytest.raises(TransientDownloadError, match="Failed to download file"):
            download_file("https://example.com/file.txt", outdir=temp_dir, overwrite=True)

        assert dest_path.read_text() == "the good copy"

    def test_no_partial_file_is_left_behind(self, temp_dir, mocker):
        """Not even under another name: a stray staging file is the next run's mystery."""
        dest_path = temp_dir / "file.txt"
        dest_path.write_text("the good copy")
        self._failing_response(mocker, self._dying_chunks())

        with pytest.raises(TransientDownloadError, match="Failed to download file"):
            download_file("https://example.com/file.txt", outdir=temp_dir, overwrite=True)

        assert self._files_ignoring_the_lock(temp_dir) == ["file.txt"]

    def test_a_first_download_that_dies_leaves_nothing(self, temp_dir, mocker):
        self._failing_response(mocker, self._dying_chunks())

        with pytest.raises(TransientDownloadError, match="Failed to download file"):
            download_file("https://example.com/file.txt", outdir=temp_dir)

        assert self._files_ignoring_the_lock(temp_dir) == []

    def test_a_successful_overwrite_still_replaces_the_file(self, temp_dir, mocker):
        dest_path = temp_dir / "file.txt"
        dest_path.write_text("the old copy")
        self._failing_response(mocker, [b"test ", b"data"])

        result = download_file("https://example.com/file.txt", outdir=temp_dir, overwrite=True)

        assert result == dest_path
        assert dest_path.read_text() == "test data"
        assert self._files_ignoring_the_lock(temp_dir) == ["file.txt"]


class TestWhichFailuresAreTransient:
    """A failure a later attempt may not meet is an ``OSError``; a client error is a ``ValueError``."""

    @staticmethod
    def _status(mocker, status: int):
        response = requests.Response()
        response.status_code = status
        error = requests.HTTPError(f"{status}", response=response)
        mocker.patch("gwmock.utils.download.requests.get", side_effect=error)

    @pytest.mark.parametrize("status", [500, 503, 408, 429])
    def test_a_server_error_or_rate_limit_is_transient(self, temp_dir, mocker, status):
        self._status(mocker, status)
        with pytest.raises(TransientDownloadError):
            download_file("https://example.com/file.txt", outdir=temp_dir)

    def test_transient_failures_are_os_errors(self) -> None:
        assert issubclass(TransientDownloadError, OSError)

    @pytest.mark.parametrize("status", [403, 404])
    def test_a_client_error_is_not(self, temp_dir, mocker, status):
        self._status(mocker, status)
        with pytest.raises(ValueError, match="Failed to download file") as excinfo:
            download_file("https://example.com/file.txt", outdir=temp_dir)
        assert not isinstance(excinfo.value, OSError)
        assert isinstance(excinfo.value, DeterministicRefusalError)
