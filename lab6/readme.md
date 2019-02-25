# Lab 6/7: Testing and Deployment

As always, the first thing to do is `git pull` :)

In this lab, we will

- Add evaluation tests
- Set up continuous integration via CircleCI, and see our commits pass/fail
- Run our LinePredictor as a web app, and send it some requests
- Dockerize our web app
- Deploy our web app as a serverless function to AWS Lambda
- Look at basic metrics and set up a more advanced one
- Experience something going wrong in our deployed service, and catching it with metrics

This lab has quite a few new files. We'll go through them in order.

## Setting up CircleCI

The relevant new files for setting up continuous integration are

- `evaluation/evaluate_character_predictor.py`
- `evaluation/evaluate_line_predictor.py`
- `tasks/run_validation_tests.sh`

There is one additional file that is outside of the lab6 directory (in the top-level directory): `.circleci/config.yml`

Let's set up CircleCI first and then look at the new evaluation files.

Go to https://circleci.com and log in with your Github account.
Click on Add Project. Select your fork of the `fsdl-text-recognizer-project` repo.
It will ask you to place the `config.yml` file in the repo.
Good news -- it's already there, so you can just hit the "Start building" button.

While CircleCI starts the build, let's look at the `config.yml` file.

Let's also check out the new validation test files: they simply evaluate the trained predictors on respective test sets, and make sure they are above threshold accuracy.

Now that CircleCI is done building, let's push a commit so that we can see it build again, and check out the nice green chechmark in our commit history (https://github.com/sergeyktest/fsdl-text-recognizer-project/commits/master)

## Serving predictions from a web server

First, we will get a Flask web server up and running and serving predictions.

```
pipenv run python api/app.py
```

Open up another terminal tab (click on the '+' button under 'File' to open the
launcher). In this terminal, we'll send some test image to the web server
we're running in the first terminal.

**Make sure to `cd` into the `lab6` directory in this new terminal.**

```
export API_URL=http://0.0.0.0:8000
curl -X POST "${API_URL}/v1/predict" -H 'Content-Type: application/json' --data '{ "image": "data:image/png;base64,'$(base64 -w0 -i text_recognizer/tests/support/emnist_lines/or\ if\ used\ the\ results.png)'" }'
```

If you want to look at the image you just sent, you can navigate to
`lab6/text_recognizer/tests/support/emnist_lines` in the file browser on the
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

Still in the `lab6` directory, run:

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

Next, run `sls info`. You'll see a message asking you to set up your AWS credentials. We sent an email to you with your AWS credentials (let us know if you can't find it).

Note that emailing credentials is a bad idea. You usually want to handle credentials in a more secure fashion.
We're only doing it in this case because your credentials give you limited access and are for a temporary AWS account.

You can also go to https://379872101858.signin.aws.amazon.com/console and log in with the email you used to register (and the password we emailed you), and create your own credentials if you prefer.

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

As before, we can test out our API by running a few curl commands (from the `lab6` directory). We need to change the `API_URL` first though to point it at Lambda:

```
export API_URL="https://REPLACE_THIS.execute-api.us-west-2.amazonaws.com/dev/"
curl -X POST "${API_URL}/v1/predict" -H 'Content-Type: application/json' --data '{ "image": "data:image/png;base64,'$(base64 -w0 -i text_recognizer/tests/support/emnist_lines/or\ if\ used\ the\ results.png)'" }'
curl "${API_URL}/v1/predict?image_url=http://s3-us-west-2.amazonaws.com/fsdl-public-assets/emnist_lines/or%2Bif%2Bused%2Bthe%2Bresults.png"
```

If the POST request fails, it's probably because you are in `api` and not in the top-level `lab6` directory.

You'll want to run the curl commands a couple of times -- the first execution may time out, because the function has to "warm up."
After the first request, it will stay warm for 10-60 minutes.

## Lambda monitoring

We're going to check the logs and set up monitoring for your deployed API. In order to make the monitoring more interesting, we're going to simulate people using your API.

**In order for us to do that, you need to go to https://goo.gl/forms/YQCXTI2k5R5Stq3u2 and submit your endpoint URL.**
It should look like this (ending in "/dev/"):
```
https://REPLACE_THIS.execute-api.us-west-2.amazonaws.com/dev/
```

If you haven't already sent a few requests to your endpoint, you should do so using the curl commands above.

Next, log in to the AWS Console at https://379872101858.signin.aws.amazon.com/console (you should've gotten an email with your username and password).

**Make sure that you switch into the Oregon region (also known as `us-west-2`) using the dropdown menu in the top right corner.**

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

You can play with your API here a bit while we turn on the traffic for everyone. Double check that you've submitted your endpoint to the Google form above.

Once the traffic is going, refresh your dashboard a bit and watch it. We're going to change something about the traffic, and it's going to start making your API perform poorly.
Try and figure out what's going on, and how you can fix it. We'll leave the adversarial traffic on for a while.

If you're curious, you can add a metric filter to show memory usage with this pattern:
```
[report_name="REPORT", request_id_name="RequestId:", request_id_value, duration_name="Duration:", duration_value, duration_unit="ms", billed_duration_name_1="Billed", bill_duration_name_2="Duration:", billed_duration_value, billed_duration_unit="ms", memory_size_name_1="Memory", memory_size_name_2="Size:", memory_size_value, memory_size_unit="MB", max_memory_used_name_1="Max", max_memory_used_name_2="Memory", max_memory_used_name_3="Used:", max_memory_used_value, max_memory_used_unit="MB"]
```

You can name it `USERNAME_memory`. Select `$max_memory_used_value` for the metric value.

Make sure to save your dashboard!
