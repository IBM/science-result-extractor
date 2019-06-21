# Downloader

download_pdfs.py allows to download the entire collection of raw PDFs of the NLP-TDMS dataset.

## Requirements

The only requirement is Python 3.

## Usage

```python download_pdfs.py```

or 

```python3 download_pdfs.py```

## Notes

The script checks the MD5 checksum of the downloaded files against checksums of the original dataset used in the experiments of our ACL paper.
When it differs, you will see a warning in the standard output, but there is likely no cause for concern as the file is probably just a slightly different PDF version of the same paper (for example, a more recent version on arvix).
The script will also warn you if the downloaded file is not a PDF, or if it is too small, in which case you will have to open the link manually in a browser and download the PDF yourself.