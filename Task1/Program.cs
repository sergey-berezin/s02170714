using System;
using ClassLibrary;

namespace Task1
{
    class Program
    {
        static void Main(string[] args)
        {
            string model_Path = "../../../../resnet50-v2-7.onnx";

            Console.WriteLine("Type the image path:");
            string image_path = Console.ReadLine();

            OnnxGeneral onnxGeneral = new OnnxGeneral(model_Path);
            Threadworks threadworks = new Threadworks(image_path, onnxGeneral);
            threadworks.Run();
            threadworks.ToString();
        }
    }
}
