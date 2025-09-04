# QA-LoRA-Lab

Repositorio con el código para reproducir experimentos de fine-tuning y evaluación de modelos (Qwen/Llama) con LoRA y QLoRA sobre el dataset de QA usado en la competencia QuALES.

## Datos
- Scripts esperan por defecto en la raíz: `dataset_covid_qa_train.json` y `dataset_covid_qa_dev_gold.json`.

## Ejecutar
Ejecutar todas las combinaciones de hiperparámetros para LoRA:
```
python .\run_lora_experiments.py
```

Ejecutar todas las combinaciones de hiperparámetros para QLoRA:
```
python .\run_qlora_experiments.py
```

## Salidas
- Métricas por corrida acumuladas: `training_metrics.csv`.
- Resultados de validación: `eval_results_dev.csv` dentro de la carpeta base de cada corrida.
- Perfiles (PyTorch Profiler + TensorBoard): carpeta `.../profiler/` de cada corrida.
- Carpeta de salida por corrida con timestamp bajo `lora/...` o `qlora/...`.

## Modelos en HuggingFace
- [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

**Importante:** Los modelos de Llama requieren aceptar los términos de uso en HuggingFace para poder descargarlos, y agregar un token de autenticación como variable de entorno con la clave `HF_TOKEN`.