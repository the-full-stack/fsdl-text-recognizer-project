# FSDL Handwriting Dataset

## Collection

Handwritten paragraphs were collected in the FSDL March 2019 class.
The resulting PDF was stored at https://fsdl-public-assets.s3-us-west-2.amazonaws.com/fsdl_handwriting_20190302.pdf

Pages were extracted from the PDF by running `gs -q -dBATCH -dNOPAUSE  -sDEVICE=jpeg -r300 -sOutputFile=page-%03d.jpg -f fsdl_handwriting_20190302.pdf` and uploaded to S3, with urls like https://fsdl-public-assets.s3-us-west-2.amazonaws.com/fsdl_handwriting_20190302/page-001.jpg
