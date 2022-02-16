import torch from '@pytorchlive/torch';
import { transforms as T } from '@pytorchlive/vision';
import media from '@pytorchlive/media';
import { Image, ModelLoader } from 'react-native-pytorch-core';
import * as CharacterClasses from './CharacterClasses.json';

// Super Mario model url
const MODEL_URL = 'https://github.com/raedle/classify-super-mario/releases/download/v0.0.1-alpha.11/super_mario.ptl';

const normalizeFunc = T.normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
const resizeFunc = T.resize(224);

// Super Mario character classifiction model
let model: any = null;

// Load TorchScript Lite Interpreter model
export async function loadModel(loaderFunc?: (url: string) => Promise<string>) {
  // Model already loaded
  if (model != null) {
    return;
  }
  let filePath: string;
  if (loaderFunc != null) {
    filePath = await loaderFunc(MODEL_URL);
  }
  else {
    filePath = await ModelLoader.download(MODEL_URL);
  }
  model = torch.jit._load_for_mobile(filePath);
}

/**
 * Classify the Super Mario character in an input image.
 *
 * @param image An input image with a Super Mario character
 * @returns The name of the Super Mario character
 */
export function classifyCharacter(image: Image) {
  if (model == null) {
    throw new Error('Model not loaded. Call "await loadModel()" function');
  }

  // Image width and height
  const width = image.getWidth();
  const height = image.getHeight();

  // Convert image to blob
  const blob = media.toBlob(image);

  // Get tensor from image blob in [H, W, C]
  let tensor = torch.fromBlob(blob, [height, width, 3]);

  // Permute to [C, H, W]
  tensor = torch.permute(tensor, [2, 0, 1]);

  // Div tensor to have values from [0, 1]
  tensor = tensor.div(255);

  // Normalize image tensor data
  tensor = normalizeFunc(tensor);

  // CenterCrop image tensor [3, min(H, W), min(H, W)]
  const centerCropFunc = T.centerCrop(
    Math.min(image.getWidth(), image.getHeight())
  );
  tensor = centerCropFunc(tensor);

  // Unsqueeze to [1, 3, min(H, W), min(H, W)]
  tensor = tensor.unsqueeze(0);

  // Resize to [1, 3, 224, 224]
  tensor = resizeFunc(tensor);

  // Run model inference
  const result = model.forward(tensor);

  // Get result [1, 5] -> 5 characters
  const resultTensor = result.toTensor();

  // Get index for character with higher probability
  const maxIdx = torch.argmax(resultTensor);

  // Return character name
  return CharacterClasses[maxIdx];
}
