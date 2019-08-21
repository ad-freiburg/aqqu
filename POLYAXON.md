# Polyaxon Training

For starting the training with polyaxon one has to do the following in advance.

## Download Polyaxon CLI

```
pip install polyaxon-cli
```

## Login to Polyaxon

```
polyaxon login [--username=...] [--password=...]
```

## Create Project

```
polyaxon project create --name='aqqu' --description='This is optinal'
```

## Initialize Polyaxon Workspace

Within this directory: 

```
polyaxon init aqqu
```

## Before Training

Adapt the paths to the resources. Currently there are two different storage places:

```/data/1/``` is located under ```titan:/local/hdd/exports/data/```

and

```/data/local/``` is located on ```[rubur,flavus]:/local/ssd/``` (so the data must be available on both machines.

## Upload and Run

```
polyaxon run -u [-f polyaxonfile.yml]
```
