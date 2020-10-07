using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.PixelFormats;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ClassLibrary
{
    public class OnnxGeneral
    {
        private InferenceSession name_model;
        public OnnxGeneral(string modelFilePath)
        {
            name_model = new InferenceSession(modelFilePath);
        }

        public static DenseTensor<float> ImageModel(string imageFilePath)
        {
            using Image<Rgb24> image = Image.Load<Rgb24>(imageFilePath);
            const int TargetWidth = 224;
            const int TargetHeight = 224;

            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(TargetWidth, TargetHeight),
                    Mode = ResizeMode.Crop
                });
            });

            var input = new DenseTensor<float>(new[] { 1, 3, TargetWidth, TargetHeight });
            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var stddev = new[] { 0.229f, 0.224f, 0.225f };
            for (int y = 0; y < TargetHeight; y++)
            {
                Span<Rgb24> pixelSpan = image.GetPixelRowSpan(y);
                for (int x = 0; x < TargetHeight; x++)
                {
                    input[0, 0, y, x] = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
                    input[0, 1, y, x] = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
                    input[0, 2, y, x] = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];
                }
            }
            return input;
        }

        public string PredictModel(DenseTensor<float> input)
        {
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(name_model.InputMetadata.Keys.First(), input)
            };

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = name_model.Run(inputs);

            var output = results.First().AsEnumerable<float>().ToArray();
            float sum = output.Sum(x => (float)Math.Exp(x));
            var max = output.Select(x => (float)Math.Exp(x) / sum);
            var Labels_indx = max.ToList().IndexOf(max.Max());
            return LabelMap.Labels[Labels_indx];
        }
    }
}
