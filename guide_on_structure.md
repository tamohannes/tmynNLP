# tmynnlp

A guide to Natural Language Processing with tmynnlp

## Table of Contents

1. Getting Started
2. Main Components
    1. Command
        1. Runner
        2. Purge
        3. Cleanup
    2. Core Components of the Runner
        1. DatasetReader
        2. Preprocessor
        3. Tokenizer
        4. FeatureExtractor
        5. Metric
        6. Experiment
3. Abstractions and Design
    1. The Framework Structure
    2. Using Config Files: FromParams
    3. Registrable
    4. Cacheable
    5. Logger
    6. TmpHandler
    7. Device
    8. Debugging

# 1. Getting Started

So what is tmynnlp and what is it designed for? tmynnlp is a Natural Language Processing specific framework that gives you a high level of abstraction for the main components in the NLP pipelines. We can run any experiment using tmynnlp. It is not deep learning specific, and one can run statistical learning and even deterministic algorithms in the scope of the tmynnlp.

In this section of the guide, we'll give a quick start on one of the most basic things you can do with tmynnlp: text/document classification, for which we already have a few implemented experiments. Let’s start with the setup steps.

First, we need to create a conda environment to keep all the necessary packages and the required versions in one place. For that, we can simply run the following command:

```bash
conda create -n tmynnlp_env python=3.8
```

---

This framework requires python3.8 or higher, so we specify that in command of environment creation. One can simply install the latest version of Python (at the time of writing this it is 3.10) and hope for the best.

After this, we need to activate the newly created environment:

```bash
conda activate tmynnlp_env
```

---

Now it’s time to install the dependencies and add tmynnlp to the list of our pip packages.

```bash
pip install -e .
```

---

To make the tmynnlp executable from any location: add this alias in your terminal config file, e.g. “.zshrc” or “.bash_profile”.

```bash
alias tmynnlp='ABSOLUTE/PATH/TO/THE/DIR/tmynnlp/__main__.py'
```

---

Where the “...” indicates the location of tmynnlp on your machine.

After completing these steps tmynnlp is ready to use. We can run:

```bash
tmynnlp --help
```

---

to see the list of available commands.

For this particular problem, we will be using the command “runner” to run our text classification experiments.

The “runner” requires a config JSON file and packages where the concrete implementations are provided.

Say our config JSON is in the runs/runs.json location, and the package we need to include is called “modules”, then the command for firing the “runner” will look like this:

```bash
tmynnlp runner runs/runs.json --include_package modules
```

---

Please note that the “--include_package” parameter can take multiple packages (a list of package names).

An example of a “runs.json” file can be found in the exact mentioned location: “runs/runs.json”. It is a list of JSON configurations of experiments that we need to run. Please see a sample “runs.json” file below.

```json
[
    {
        "type": "exp2",
        "dataset_reader": {
            "type": "email-body",
            "train_data_path": "./data/email_bodies_data/train.json",
            "val_data_path": "./data/email_bodies_data/val.json"
        },
        "preprocessor": {
            "type": "default"
        },
        "tokenizer": {
            "type": "huggingface_tokenizer",
            "padding": true,
            "truncation": true
        },
        "feature_extractor": {
            "type": "huggingface_model",
            "pretrained_model": "bert-base-uncased"
        },
        "metrics": [
            {
                "type": "accuracy"
            },
            {
                "type": "f1",
                "average": "weighted"
            }
        ],
        "dist_metric": "cosine"
    }
]
```

---

In this particular example, we are running the experiment “exp2” which is located at the “modules/experiments/exp2.py” file. The class of “exp2” is named “Exp2”, and this requires instances of: “dataset_reader”, “preprocessor”, “tokenizer”, “feature_extractor”, and “metrics” as parameters. Those instances are connected to the “Experiment” class. The “dataset_reader”, “preprocessor”, and “metrics” are mandatory for an “Experiment” to run. This particular experiment has also extra arguments, which are only specific to it e.g. “dist_metric” and “batch_size”.

You can also notice that the config file has dependencies. For example, the “dataset_reader” which is required by the “Experiment” requires “train_data_path”, “val_data_path” and the type of the particular implementation of the “dataset_reader” which in this case is “email-body”.

More about the design of the framework and the dependency handling in Section 3.

Now, let’s run the command and see the scored “accuracy” and “f1” metrics in the output.

Congratulations, you have just run your very first experiment using tmynnlp!

Now let’s dive deeper into the components and see what each of them does.

# 2. Main Components

tmynnlp includes NLP-specific abstractions, which build the general information flow of the pipeline.

In this section, we will discuss the main component, one of its implementations. Then we will take a look at the main components' implementations, their structure, and the relationships they have with the other components. The high-level structure of the framework is presented in Figure 1.

<!-- TODO add the figure -->

Figure 1: The solid lines indicate the access. A one-sided arrow with a solid line means that the starting component has access to the ending point. The two-sided solid line shows the pointwise access. The dashed arrowed lines indicate the data flow. The components with dashed borders show that the component is optional, while the ones with solid borders are required.

The entire framework starts running from the main component: “Command”. The “Command” has multiple sub-commands or implementations. In this Figure, the “Command” fires one of its sub-commands: “runner”. This requires its specific sub-arguments: a config JSON file, packages to be included, and directories for log, cache, and temporary file storing. The last 3 parameters have default values: “./log”, “./cache” and “./tmp” respectively.

The “Experiment” is the central component of the sub-command “runner”.It has access to the main components, and vice versa. The Experiment and the components are encapsulated inside the experiment. The “Cacher”, “Logger” and the “Temporary File Handler” have access to the experiment.

## Command

The framework fires from the “Command”, which is an abstraction of commands. The “Command” parses the arguments by calling the “add_subparser” method and runs the particular command.

One can implement a command and execute a module with a specific logic by creating a file in the “commands” directory, inheriting the class from the “Command” and registering with a specific name.

## Runner

The command for running the experiments is the “runner”. This is simply designed to run (train and evaluate or only evaluate) all the specified experiment instances.

It also has a method for doing a basic setup of logging, i.e. creating a logging file with a specific name.

## Prune

This command is designed to prune/remove some of the elements from the specified directories. The specified directories list should be a sublist of the: [“cache”, “tmp” ,“log”] list. The “prune” command works with the LRU (Least Recently Used) cache logic. It runs recursively in the specified directory, finds all of the files, takes their least recently accessed times, and sorts them. After which keep the least recently used “n” files. The “n” by default is 10, one can set the desired value by specifying the value of the argument “--num_of_elements”.

## Cleanup

The command “cleanup” is similar to “prune”. Unlike the “prune”, this simply removes everything in the specified directories. The directories list should again be a sublist of the: [“cache”, “tmp” ,“log”] list.

## Core Components of the Runner

In this subsection, we will take a closer look at the design of the “Runner” command. Below are presented the core components. The “DatasetReader”, “Preprocessor”, “Tokenizer”, “FeatureExtractor”, “Metric”, and the “Experiment”. All of the core components except the “Experiment” are usually called by the same sequence as listed above or as it is shown in Figure 1. The “Experiment” is the main component where the rest of the components are used. Note that the “Tokenizer” and “FeatureExtractor” are not mandatory components.

An experiment can receive multiple of the same type of component. For example, one might need to use two different “Preprocessor”s for some reason. In that case, that particular implementation of an experiment needs to receive the instance of the second “Preprocessor” in the constructor method of the class.

## DatasetReader

What do we need for an experiment? Right, data! To read our data and make it comfortable to work with we need to use the “DatasetReader” component. Which will read the data from any given source e.g. JSON, database, CSV file, and will return a dictionary consisting of two elements: the training and the validation “Dataset”s. The “Dataset” class is the “datasets.Dataset” class from “huggingface.datasets”, which inherits from “pyarrow.Table”. The “DatasetReader” returns a “datasets.DatasetDict” which contains two keys: train and val and the corresponding “Dataset” instances.

The “DatasetReader” has two mandatory methods: “read” and “_read”, the first one is public, and the second one is for loading the data under the hood and passing it to the first method so it will return the dictionary of “Dataset”s to the user.

The “DatasetReader” has one more parameter: “mock”, a boolean parameter. If this parameter is true then the “DatasetReader” will return only the first “n” samples, it is for testing purposes only, to quickly run the processes and see if the entire pipeline has some errors. By default it is false.

## Preprocessor

The “Preprocessor” is designed to do basic cleaning on the raw dataset which is loaded by the “DatasetReader”.

Inside the “__call__” method one can implement a dataset specific sequence of operations, such as: input data trimming, manipulations, removing empty or invalid instances, cleaning some extra characters, and so on.

## Tokenizer

As the name suggests, the “Tokenizer” does tokenization, it overrides the “__call__” method and also has an optional method: “_atomic”, which is designed to handle the tokenization of atomic elements (a batch or a single sample). The “__call__” itself might receive a list of inputs that we might need to batchify to process. Note that the “Tokenizer” is not a mandatory component in the pipeline.

## FeatureExtractor

This abstraction is usually the next step of “Tokenizer” in the pipeline. The “FeatureExtractor” calls a pre-trained model and runs it in the inference mode by giving the provided input. On the output, we usually expect a tensor or a representation of the given input in the form of embeddings. Note that the pooling is not done in this scope.

Note: do not forget to turn off the calculation of the gradient when using a pre-trained LM in inference mode.

## Metric

The “Metric” itself does a simple calculation and returns the particular metric. In the “Experiment” we expect a list of “Metric”s. The “Metric”s are called in the end, when the experiment returns two lists: one with the predicted labels (“predictions”) and the other one with the gold labels (“gold_labels”). So this output is given to each “Metric”, and the latter returns a dict with metric names and their corresponding scores. This also returns the number of samples for which the experiment has no predictions - n_no_predictions.

## Experiment

The “Experiment” is the most flexible component, also the component where most of the logic should be written. It takes all of the previously mentioned component instances as an argument, plus the number of processes/workers in case of using multiprocessing/parallelism (“num_workers”).

This abstraction has 3 main methods: “info”, “score” and “__call__”. The first one simply returns a string with some basic info about the experiment. The score method is inherited from the parent “Experiment”. It takes care of calling the list of “Metric”s and returns a dict, with the metric names and the corresponding scores.

The “__call__” is the central method.

In the “__call__” the user calls all the components mentioned previously, and connects one’s output with another’s input by creating the desired pipeline. There is one requirement on the “__call__” : it must return the predicted labels/values and the target labels/values lists.

The “Experiment” has access to all of its dependencies which inherit from the “Cacheable” and vice versa: every component that depends on “Experiment” can access the latter by calling the “_parent” property. This reverse access is created with the help of the “reverse_registration” method.

DO NOT FORGET to create properties (encapsulate) for your custom parameters that you receive. In this way, tmynnlp will keep track of your particular experiment and the caching mechanism will work as expected.

# 3. Abstractions and Design

A core motivating principle of tmynnlp (and object-oriented software design generally) is to separate the configuration of an object from its implementation code. This is accomplished with dependency injection, which is a fancy term that in practice just means “objects take all of their instance variables as constructor parameters”.

We write our code using high-level abstractions, which lets us make changes to lower-level details by just changing the constructor parameters that we pass to the main components. It's the job of the final script that runs things to configure the behavior of the code by creating all of the objects that get passed to these constructors.

## The Framework Structure

Here we will take a closer look at the structure of the framework and each directory meaning. The main directory tree is presented in Figure 2.

```txt
tmynnlp
├── README.md
├── cache
│   └── ...
├── data
│   ├── document_data
│   │   ├── ...
│   └── email_bodies_data
│       ├── ...
├── helpers
│   └── dataset_spliter.py
├── logs
│   └── ...
├── runs
│   ├── runs.json
├── setup.py
├── tmp
│   └── ...
├── tmynnlp
│   ├── __main__.py
│   ├── commands
│   │   ├── __init__.py
│   │   ├── command.py
│   │   └── runner.py
│   ├── common
│   │   ├── __init__.py
│   │   ├── cacheable.py
│   │   ├── from_params.py
│   │   ├── lazy.py
│   │   ├── params.py
│   │   ├── registrable.py
│   │   └── util.py
│   ├── cores
│   │   ├── __init__.py
│   │   ├── dataset_reader.py
│   │   ├── experiment.py
│   │   ├── feature_extractor.py
│   │   ├── metric.py
│   │   ├── preprocessor.py
│   │   └── tokenizer.py
│   └── modules
│       ├── ...
└───
```

---

Figure 2: The recursive tree representation of the main directory. The directories colored with: yellow - are created during the execution, and dark gray - are optional.

- README.md - the installation steps and some examples are given.
- setup.py - installs the required packages for the framework and registers the framework in the pip packages list.
- cache, logs, and tmp directories - the corresponding files/directories are stored.
- runs directory - the config JSON files can be saved here - optional.
- data directory - the data files can be stored - optional.
- helpers directory - is not a part of the framework, one can put helpful scripts such as splitting the data into training and validation sets.
- tmynnlp directory - this is the main directory of the framework.
    - __main__.py - fires the framework.
    - commands module - handles the argument reading and calls the required subcommand.
    - common module - contains some core classes which the main modules can extend to enlarge their scope of action. Also contains a util.py file where some general functions can be found.
    - cores module - provides all the abstractions/main-components.
    - modules module - contains the implementations of the abstractions declared in cores. This is not a mandatory dir of the framework, one can create this outside of the parent dir.

A sample structure of the “modules” directory is presented in Figure 3.

```txt
modules
├── __init__.py
├── dataset_readers
│   ├── __init__.py
│   ├── document.py
│   └── email_body.py
├── experiments
│   ├── __init__.py
│   ├── exp1.py
│   ├── exp2.py
│   ├── exp3.py
├── feature_extractors
│   ├── __init__.py
│   ├── doc2vec.py
│   ├── huggingface_model.py
│   ├── huggingface_ner.py
│   └── huggingface_pipeline.py
├── metrics
│   ├── __init__.py
│   ├── accuracy.py
│   └── f1.py
├── preprocessors
│   ├── __init__.py
│   ├── default.py
│   └── email_body.py
└── tokenizers
    ├── __init__.py
    ├── huggingface_tokenizer.py
    └── nltk_tokenizer.py
```

---

Figure 3: A sample modules directory tree representation.

## Using Config Files: FromParams

Our framework, based on the “FromParams” class, is very simple. All we do is match type annotations on constructor arguments to parameter dictionaries loaded from a JSON file. That's it. Its utility comes from handling parameterized collections, user-defined types, polymorphism, and operating recursively.

Under the hood, all that “FromParams” is doing is constructing arguments for an object from JSON, but, once you get the hang of it, this simple functionality enables you to write and modify reproducible experiments with ease.

To construct a class from a JSON dictionary, it has to inherit from the “FromParams” abstract base class.

The “.from_params()” method (provided by “FromParams”) looks in the parameter dictionary and matches keys in that dictionary to argument names in the “Class” constructor (by default the constructor of the class is called, one can change it by specifying “default_implementation” during the registration). When it finds a match, it looks at the type annotation of the argument in the constructor and constructs an object of the appropriate type from the corresponding value in the parameter dictionary.

But “FromParams” can do much more than pull parameters out of dictionaries; it can also handle user-defined types, which are constructed recursively. For example, we can use our “HuggingFaceTokenizer” class as a type annotation on another class that inherits from “FromParams” for example “HuggingFaceNer” (See in the modules directory).

In practice, what we do in tmynnlp is have all of our classes inherit from “FromParams”, from our main components to our “Experiment” itself, so that we can have one file that specifies all of the configurations that goes into an entire experiment. Everything is built recursively from the specification in a file that looks something like this:

```json
[ 
     {
        "ner_extractor": {
            "type": "huggingface_ner",
            "pretrained_model": "dslim/bert-base-NER",
            "ner_tokenizer": {
                "type": "huggingface_tokenizer",
                "pretrained_model": "bert-base-cased"
            }
        },
    }
]
```

---

## Registrable

Another fundamental design principle in tmynnlp is the use of polymorphism to abstract away low-level details of data processing or model operations. We encapsulate common operations in abstract base classes, then write code using those abstract base classes instead of concrete instantiations.

Using these abstract base classes as type annotations presents a challenge to constructing objects using “FromParams”—we need a way of knowing which concrete instantiation to construct, from the configuration file. We handle this sort of polymorphic dependency injection using a class called “Registrable”.

When we create an abstract base class, like “Preprocessor”, we have it inherited from “Registrable”. Concrete subclasses then use a decorator provided by “Registrable” to make themselves known to the base class: “@Preprocessor.register("my_preprocessor_name")”. Then, when we are trying to construct an argument with a type annotated with the base “Preprocessor” class, “FromParams” looks for a type key whose value must correspond to some registered name.

## Cacheable

Most of the “Experiments” require some heavy computations, which are not only time but resource consuming. To not calculate the same things more than one we can use the “Cacheable” static class. Which has a main static method called: “cache”, which plays the role of a proxy. So it stops the execution of a method and checks in the cache if that particular method with the specified parameters and with the preceding modules with their parameters appear in the cache. If that’s the case it loads the pickle file, otherwise, it continues the method execution and in the end, it gets the output of the method and dumps a pickle file in the cache. It is worth mentioning that the caches are shareable across the “Experiment”s.

All you need to do to use the “cache” method, is inherit from the Cacheable method add a decorator “@Class.cache()” on the particular method whose output you want to be cached and that’s it.

Note that the “Cachable” ignores the “batch_size” parameter, so multiple exact runs with different batches will have one single cache. One can modify the list of such parameters by appending the required keys to the “Cachable.exceptional_properties” property.

The “Cacheable.cache()” method also logs information about whether it reads from a cache file or actually executes the given method. The default logging behavior of the “cache()” method is described by its nature. Methods which usually take a lot of computational resources and time, are cached, thus it is logical to track their state via a simple logging by default.

## Logger

The “Logger” does the config setup of the already existing Python “logging” package - logging_config(). The actual logging can be done from any location of the framework by using the “logging” module.

## TmpHandler

This simple static class is designed to handle temporary file operations, such as: generating a path for a given module and arguments - get_path(), checking weather a given path exists or not - exists(), strong a temporary file in a specific directory - store(), and getting a tmp file - get().

## Device

The “Device” class is a trivial static class. Which only has one class property: “device”, which is a torch.device type variable, and indicates the currently active device. This is used to transfer the models to the device in the specific implementation, e.g. “HuggingFaceModel”. One can use this to handle the cloud GPU connections in the future.

## Debugging

One way of checking what’s going on on the system is to take a look at the logs. tmynnlp will create a log file for each execution of a config file, in the “logs” directory, by specifying the execution start time in the name of the log file. In which you can see the logs of different processes and their time and even the exceptions that you might get in the runtime.

One can create an abstraction for logging mechanisms. One of the potential implementations of this abstraction will not only print logs in local files but can also send the status/current state of the system via an email or using a web API.

Despite this, modern IDEs such as “Visual Studio Code” or “PyCharm” can be used in the debug mode to see the process of the code execution. It’s super easy to debug tmynnlp once you get hang of the previous sections.