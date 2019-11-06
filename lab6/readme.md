# Lab 6: Data Labeling and Versioning

In this lab we will annotate the handwriting samples we collected, export and version the resulting data, write an interface to the new data format, and download the pages in parallel.

## Data labeling

We will be using a simple online data annotation web service called Dataturks.

Please head to the [project page](https://dataturks.com/projects/sergeykarayev/fsdl_handwriting) and log in using our shared credential: `annotator@fullstackdeeplearning.com` (the password will be shared during lab).

You should be able to start tagging now.
Let's do it together for a little bit, and then you'll have time to do a full page by yourself.

We'll sync up and review results in a few minutes.

(Review results and discuss any differences in annotation and how they could be prevented.)

## Export data and update metadata file

Let's now export the data from Dataturks and add it to our version control.

You have noticed the `metadata.toml` files in all of our `data/raw` directories.
They contain the remote source of the data, the filename it should have when downloaded, and a SHA-256 hash of the downloaded file.

The idea is that the data file has all the information needed for our dataset.
In our case, it has image URLs and all the annotations we made.
From this, we can download the images, and transform the annotation data into something usable by our training scripts.
The hash, combined with the state of the codebase (tracked by git), then uniquely identifies the data we're going to use to train.

We replace the current `fsdl_handwriting.json` with the one we just exported, and now need to update the metadata file, since the hash is different.
SHA256 hash of any file can be computed by running `shasum -a 256 <filename>`.
We can also update `metadata.toml` with a convenient script that replace the SHA-256 of the current file with the SHA-256 of the new file.
There is a convenience task script defined: `tasks/update_fsdl_paragraphs_metadata.sh`.

The data file itself is checked into version control, but tracked with git-lfs, as it can get heavyweight and can change frequently as we keep adding and annotating more data.
Note that `git-lfs` actually does something very similar to what we more manually do with `metadata.toml`.
The reason we also use the latter is for standardization across other types of datasets, which may not have a file we want to check into even `git-lfs` -- for example, EMNIST and IAM, which are too large as they include the images.

## Download images

The class `IamHandwritingDataset` in `text_recognizer/datasets/iam_handwriting.py` must be able to load the data in the exported format and present it to consumers in a format they expect (e.g. `dataset.line_regions_by_id`).

Since this data export does not come with images, but only pointers to remote locations of the images, the class must also be responsible for downloading the images.

In downloading many images, it is very useful to do so in parallel.
We use the `concurrent.futures.ThreadPoolExecutor` method, and use the `tqdm` package to provide a nice progress bar.

## Looking at the data

We can confirm that we loaded the data correctly by looking at line crops and their corresponding strings.

Make sure you are in `lab6` directory, and take a look at `notebooks/05-look-at-fsdl-handwriting.ipynb`.

## Training on the new dataset

We're not going to have time to train on the new dataset, but that is something that is now possible.
As an exercise, you could write `FsdlHandwritingLinesDataset` and `FsdlHandwritingParagraphsDataset`, and be able to train a model on a combination of IAM and FSDL Handwriting data on both the line detection and line text prediction tasks.
