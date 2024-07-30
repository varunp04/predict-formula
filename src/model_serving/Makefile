IMAGE_NAME=formula-1-prediction
HISTORY_FILE=.image_bash_history

build:
	docker build --no-cache -t $(IMAGE_NAME) .

run:
	
	touch $(PWD)/$(HISTORY_FILE)
	docker run -it \
			   --rm \
			   --entrypoint=bash \
			   -e HISTFILE=/home/root/$(HISTORY_FILE) \
			   -v $(PWD)/$(HISTORY_FILE):/app/$(HISTORY_FILE) \
			   -v $(PWD)/:/home/root/:rw \
			   -v $(PWD)/data/iris_data.csv:/opt/ml/input/data/training/iris_data.csv \
			   -p 8501:8501 \
			   $(IMAGE_NAME)