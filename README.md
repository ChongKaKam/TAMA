# See it, Think it, Sorted: Large Multimodal Models are Few-shot Time Series Anomaly Analyzers

## Get started

### 1. Environment Setup
```shell
pip install -r requirements.txt
```

### 2.Add your API keys 

Before you get started, it is necessary to get API keys of LLMs. In our framework, you should create a .yaml file called `api_keys.yaml` in `BigModel/` directory. The format is shown below:

```yaml
openai:
  api_key: 'Your API Keys'
chatglm:
  api_key: 'Your API Keys'
```

### 3. Run
```shell
# convert sequence data into image
python3 make_dataset.py --dataset UCR --mode train --modality image --window_size 600 --stride 200
python3 make_dataset.py --dataset UCR --mode test --modality image --window_size 600 --stride 200
# convert sequence data into text
python3 make_dataset.py --dataset UCR --mode train --modality text --window_size 600 --stride 200
python3 make_dataset.py --dataset UCR --mode test --modality text --window_size 600 --stride 200

# Image-modality
python3 main_cli.py --dataset UCR --normal_reference 3 --LLM 'GPT-4o'
# Text-modality
python3 main_cli_text.py --dataset UCR --normal_reference 1 --LLM 'GPT-4o'
```

### 4. Results Analysis
```shell
# evaluation
python3 evaluation.py
# ablation study
python3 ablation_eval.py
```