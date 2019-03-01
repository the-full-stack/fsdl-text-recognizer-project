# IAM Dataset

The IAM Handwriting Database contains forms of handwritten English text which can be used to train and test handwritten text recognizers and to perform writer identification and verification experiments.

- 657 writers contributed samples of their handwriting
- 1,539 pages of scanned text
- 13,353 isolated and labeled text lines

- http://www.fki.inf.unibe.ch/databases/iam-handwriting-database

## Pre-processing

First, all forms were placed into one directory called `forms`, from original directories like `formsA-D`.

To save space, I converted the original PNG files to JPG, and resized them to half-size
```
mkdir forms-resized
cd forms
ls -1 *.png | parallel --eta -j 6 convert '{}' -adaptive-resize 50% '../forms-resized/{.}.jpg'
```

## Split

The data split we will use is
IAM lines Large Writer Independent Text Line Recognition Task (lwitlrt): 9,862 text lines.

- The validation set has been merged into the train set.
- The train set has 7,101 lines from 326 writers.
- The test set has 1,861 lines from 128 writers.
- The text lines of all data sets are mutually exclusive, thus each writer has contributed to one set only.
