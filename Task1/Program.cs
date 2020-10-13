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
            
            var Wait_for_enter = new Thread(new ThreadStart(() =>
            {
                while (Console.ReadKey().Key != ConsoleKey.Enter) { }
                Threadworks.cancelTokenSource.Cancel();
            }));
            Wait_for_enter.Start();            
            
            threadworks.Run();
            threadworks.ToString();
            
            System.Environment.Exit(0);
        }
    }
}
