# Lab 9: Web Deployment

In this lab, we will

- Run our LinePredictor as a web app, and send it some requests
- Dockerize our web app
- Deploy our web app as a serverless function to AWS Lambda
- Look at basic metrics and set up a more advanced one
- Experience something going wrong in our deployed service, and catching it with metrics

This lab has quite a few new files, mostly in the new `api/` directory.

## Serving predictions from a web server

First, we will get a Flask web server up and running and serving predictions.

```
pipenv run python api/app.py
```

Open up another terminal tab (click on the '+' button under 'File' to open the
launcher). In this terminal, we'll send some test image to the web server
we're running in the first terminal.

**Make sure to `cd` into the `lab9` directory in this new terminal.**

```
export API_URL=http://0.0.0.0:8000
curl -X POST "${API_URL}/v1/predict" -H 'Content-Type: application/json' --data '{ "image": "data:image/png;base64,'$(base64 -w0 -i text_recognizer/tests/support/emnist_lines/or\ if\ used\ the\ results.png)'" }'
```

If you want to look at the image you just sent, you can navigate to
`lab9/text_recognizer/tests/support/emnist_lines` in the file browser on the
left, and open the image.

We can also send a request specifying a URL to an image:
```
curl "${API_URL}/v1/predict?image_url=http://s3-us-west-2.amazonaws.com/fsdl-public-assets/emnist_lines/or%2Bif%2Bused%2Bthe%2Bresults.png"
```

You can shut down your flask server now.

## Adding web server tests

The web server code should have a unit test just like the rest of our code.

Let's check it out: the tests are in `api/tests/test_app.py`.
You can run them with

```sh
tasks/test_api.sh
```

## Running web server in Docker

Now, we'll build a docker image with our application.
The Dockerfile in `api/Dockerfile` defines how we're building the docker image.

Still in the `lab9` directory, run:

```sh
tasks/build_api_docker.sh
```

This should take a couple of minutes to complete.

When it's finished, you can run the server with

```sh
docker run -p 8000:8000 --name api -it --rm text_recognizer_api
```

You can run the same curl commands as you did when you ran the flask server earlier, and see that you're getting the same results.

```
curl -X POST "${API_URL}/v1/predict" -H 'Content-Type: application/json' --data '{ "image": "data:image/png;base64,'$(base64 -w0 -i text_recognizer/tests/support/emnist_lines/or\ if\ used\ the\ results.png)'" }'

curl "${API_URL}/v1/predict?image_url=http://s3-us-west-2.amazonaws.com/fsdl-public-assets/emnist_lines/or%2Bif%2Bused%2Bthe%2Bresults.png"
```

If needed, you can connect to your running docker container by running:

```sh
docker exec -it api bash
```

You can shut down your docker container now.

We could deploy this container to, for example, AWS Elastic Container Service or Kubernetes.
Feel free to do that as an exercise after the bootcamp!

In this lab, we will deploy the app as a package to AWS Lambda.

## Lambda deployment

To deploy to AWS Lambda, we are going to use the `serverless` framework.

First, let's go into the `api` directory and install the dependencies for serverless:

```sh
cd api
npm install
```

Next, we'll need to configure serverless. Edit `serverless.yml` and change the service name on the first line (you can use your Github username for USERNAME):

```
service: text-recognizer-USERNAME
```

Next, run `sls info`.
You'll see a message asking you to set up your AWS credentials.

You won't be able to quickly get those during lab right now, but you can sign for an AWS account, and note down your access key and secret key -- I store mine in 1Password, right next to my password and 2FA.

Edit the command below and substitute your credentials for the placeholders:

```
sls config credentials --provider aws --key AKIAIOSFODNN7EXAMPLE --secret wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

Now you've got everything configured, and are ready to deploy. Serverless will package up your flask API before deploying it.
It will install all of the python packages in a docker container that matches the environment lambda uses, to make sure the compiled code is compatible.
This will take 3-5 minutes. This command will package up and deploy your flask API:

```
pipenv run sls deploy -v
```

Near the end of the output of the deploy command, you'll see links to your API endpoint. Copy the top one (the one that doesn't end in `{proxy+}`).

As before, we can test out our API by running a few curl commands (from the `lab9` directory). We need to change the `API_URL` first though to point it at Lambda:

```
export API_URL="https://REPLACE_THIS.execute-api.us-west-2.amazonaws.com/dev/"
curl -X POST "${API_URL}/v1/predict" -H 'Content-Type: application/json' --data '{ "image": "data:image/png;base64,'$(base64 -w0 -i text_recognizer/tests/support/emnist_lines/or\ if\ used\ the\ results.png)'" }'
curl "${API_URL}/v1/predict?image_url=http://s3-us-west-2.amazonaws.com/fsdl-public-assets/emnist_lines/or%2Bif%2Bused%2Bthe%2Bresults.png"
```

If the POST request fails, it's probably because you are in `api` and not in the top-level `lab9` directory.

You'll want to run the curl commands a couple of times -- the first execution may time out, because the function has to "warm up."
After the first request, it will stay warm for 10-60 minutes.

## Monitoring

We can look at the requests our function is receiving in the AWS CloudWatch interface.
It shows requests, errors, duration, and some other metrics.

What it does not show is stuff that we care about specifically regarding machine learning: data and prediction distributions.

This is why we added a few extra metrics to `api/app.py`, in `predict()`.
Using these simple print statements, we can set up CloudWatch metrics by using the Log Metrics functionality.

### Log Metrics

Log in to your AWS Console, and make sure you're in the `us-west-2` region.

Once you're in, click on 'Services' and go to 'CloudWatch' under 'Management Tools.' Click on 'Logs' in the left sidebar. This will have several log groups -- one for each of us.
You can filter for yours by entering `/aws/lambda/text-recognizer-USERNAME-dev-api` (you need to enter the whole thing, not just your username).
Click on yours. You'll some log streams. If you click on one, you'll see some logs for requests to your API. Each log entry starts with START and ends with REPORT. The REPORT line has some interesting information about the API call, including memory usage and duration.

We're also logging a couple of metrics for you: the confidences of the predictor and the mean intensities of the input images.
Next, we're going to make it so you can visualize these metrics. Go back to the list of Log Groups by clicking on Logs again in the left sidebar.
Find your log group, but don't click on it. You'll see a column that says 'Metric Filters.' You currently likely have 0 filters. Click on "0 filters."
Click on 'Add Metric Filter.'

Now, we need to add a pattern for parsing our metric out of the logs. Here's one you can use for the confidence levels. Enter this in the 'Filter Pattern' box.
```
[level=METRIC, metric_name=confidence, metric_value]
```
Click on 'Assign Metric.'
Now, we need to name the metric and tell it what the data source is. Enter 'USERNAME_confidence' in the 'Metric name' box (replace USERNAME as usual). Click on 'Show advanced metric settings,' and for Metric Value, click on $metric_value to populate the text box. Hit 'Create Filter.'
Since we're already here, let's go ahead and make another metric filter for the mean intensity. You can use this Filter Pattern:
```
[level=METRIC, metric_name=mean_intensity, metric_value]
```
You should name your metric "USERNAME_mean_intensity."

Now we have a couple of metric filters set up.
Unfortunately, Metric Filters only apply to new log entries, so go back to your terminal and send a few more requests to your endpoint.

Now we can make a dashboard that shows our metrics. Click on 'Dashboards' in the left sidebar. Click 'Create Dashboard.' Name your dashboard your USERNAME.

We're going to add a few widgets to your dashboard. For the first widget, select 'Line'. In the search box, search for your username.
Click on 'Lambda > By Function Name' in the search results, and select the checkbox for 'Invocations.' This'll make a plot showing you much your API is being called.

Let's add another widget -- select Line again. Go back to the Lambda metrics and select 'Duration' this time.

Lastly, let's plot our custom metrics. Add one more 'Line' widget, search for your username again, and click on 'LogMetrics' and then 'Metrics with no dimensions'.
Check two checkboxes: `USERNAME_confidence` and `USERNAME_mean_intensity.` Before hitting Create, click on the 'Graphed Metrics' tab above, and under the 'Y Axis' column,
select the right arrow for one of the metrics (it doesn't matter which one). Now hit create.

Feel free to resize and reorder your widgets.

Make sure to save your dashboard -- else it won't persist across sessions.
