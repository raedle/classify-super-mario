import torch from '@pytorchlive/torch';
import { transforms as T } from '@pytorchlive/vision';
import media from '@pytorchlive/media';
import { Image, ModelLoader } from 'react-native-pytorch-core';
import * as CharacterClasses from './CharacterClasses.json';

// Super Mario model url
const MODEL_URL = 'https://github.com/raedle/classify-super-mario/releases/download/v0.0.1-alpha.11/super_mario.ptl';

const normalizeFunc = T.normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
const resizeFunc = T.resize(224);

let model: any = null;
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
  model = await torch.jit._load_for_mobile_async(filePath);
}

async function classifyImageInternal(image: Image, func: (tensor: any) => Promise<any>) {
  const width = image.getWidth();
  const height = image.getHeight();

  // Convert image to blob
  const blob = media.toBlob(image);

  // Get tensor from image blob
  let tensor = torch.fromBlob(blob, [height, width, 3]); // [HWC]

  // Permute
  tensor = torch.permute(tensor, [2, 0, 1]); // [CHW]

  // Div
  tensor = tensor.div(255);

  // Normalize
  tensor = normalizeFunc(tensor);

  // CenterCrop
  const centerCropFunc = T.centerCrop(
    Math.min(image.getWidth(), image.getHeight())
  );
  tensor = centerCropFunc(tensor);

  // Unsqueeze
  tensor = tensor.unsqueeze(0); // [1, 3, 480, 480]

  // Resize
  tensor = resizeFunc(tensor);

  const result = await func(tensor);
  const resultTensor = result.toTensor();
  const maxIdx = torch.argmax(resultTensor);

  return CharacterClasses[maxIdx];
}

export async function classifyCharacterSync(image: Image) {
  if (model == null) {
    throw new Error("model not loaded");
  }
  return classifyImageInternal(image, async(tensor: any) => {
    return model.forward(tensor);
  });
}

export async function classifyCharacter(image: Image) {
  if (model == null) {
    throw new Error("model not loaded");
  }
  return classifyImageInternal(image, async (tensor: any) => {
    return model.forwardAsync(tensor);
  });
}
