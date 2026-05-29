import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")

model_path = r"C:\Users\Sira\Escritorio\predapProject\transformer_outputs\models_covid_token\models\M54_base_transformer_7fh_128ff_7lb_1e-05lr.keras"

try:
    print("Loading model...")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("✅ SUCCESS: Model loaded!")
    
    if hasattr(model, 'input'):
        if isinstance(model.input, list):
            print(f"Model inputs: {len(model.input)} inputs")
            for i, inp in enumerate(model.input):
                print(f"  Input {i}: {inp.shape}")
        else:
            print(f"Input shape: {model.input.shape}")
    
    if hasattr(model, 'output'):
        if isinstance(model.output, list):
            print(f"Model outputs: {len(model.output)} outputs")
        else:
            print(f"Output shape: {model.output.shape}")
    
    print(f"Parameters: {model.count_params():,}")
    print(f"Layers: {len(model.layers)}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")