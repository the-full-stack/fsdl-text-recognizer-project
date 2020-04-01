# Lab 8: Web Deployment

## Goal of the lab

- Run our LinePredictor as a web app, and send it some requests
- Dockerize our web app
- Deploy our web app to production

## Follow along

```
git pull
cd lab8/
```

This lab has quite a few new files, mostly in the new `api/` directory.

## Serving predictions from a web server

First, we will get a Flask web server up and running and serving predictions.

```
python api/app.py
```

Open up another terminal tab (click on the '+' button under 'File' to open the
launcher). In this terminal, we'll send some test image to the web server
we're running in the first terminal.

**Make sure to `cd` into the `lab8` directory in this new terminal.**

```
export API_URL=http://0.0.0.0:8000
curl -X POST "${API_URL}/v1/predict" -H 'Content-Type: application/json' --data '{ "image": "data:image/png;base64,'$(base64 -w0 -i text_recognizer/tests/support/emnist_lines/or\ if\ used\ the\ results.png)'" }'
```

If you want to look at the image you just sent, you can navigate to
`lab8/text_recognizer/tests/support/emnist_lines` in the file browser on the
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

Still in the `lab8` directory, run:

```sh
tasks/build_api_docker.sh
```

This should take a couple of minutes to complete.

When it's finished, you can run the server with `tasks/run_api_docker.sh`


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

We could deploy this container to a number of platforms.
In this lab, we will deploy the app as a Docker container using https://render.com

## Web deployment

TODO: render.com

As before, we can test out our API by running a few curl commands (from the `lab8` directory). We need to change the `API_URL` first though to point it at Lambda:

```
export API_URL="https://REPLACE_THIS.execute-api.us-west-2.amazonaws.com/dev/"
curl -X POST "${API_URL}/v1/predict" -H 'Content-Type: application/json' --data '{ "image": "data:image/png;base64,'$(base64 -w0 -i text_recognizer/tests/support/emnist_lines/or\ if\ used\ the\ results.png)'" }'
curl "${API_URL}/v1/predict?image_url=http://s3-us-west-2.amazonaws.com/fsdl-public-assets/emnist_lines/or%2Bif%2Bused%2Bthe%2Bresults.png"
```

If the POST request fails, it's probably because you are in `api` and not in the top-level `lab8` directory.

You'll want to run the curl commands a couple of times -- the first execution may time out, because the function has to "warm up."
After the first request, it will stay warm for 10-60 minutes.
