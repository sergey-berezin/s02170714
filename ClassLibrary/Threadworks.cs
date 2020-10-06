using System;
using System.IO;
using System.Collections.Concurrent;
using System.Threading;

namespace ClassLibrary
{
    public class Threadworks
    {
        private static ManualResetEvent wait = new ManualResetEvent(false);
        private ConcurrentQueue<string> path_Imgs;

        private OnnxGeneral model;
        public int error = 0;
        public static int stop = 0;

        public Threadworks(string path, OnnxGeneral onnxModel)
        {
            model = onnxModel;

            if (Directory.Exists(path) == false)
            {
                Console.WriteLine("It is a wrong images path !");
                error = 1;
            }
            else
                path_Imgs = new ConcurrentQueue<string>(Directory.GetFiles(path, "*.jpeg"));
        }

        public void Run()
        {
            Console.WriteLine("Press Ctrl+C to stop threads ...");

            Console.CancelKeyPress += new ConsoleCancelEventHandler(myHandler);

            Thread[] ths = new Thread[Environment.ProcessorCount];

            for (int i = 0; i < Environment.ProcessorCount; ++i)
            {

                ths[i] = new Thread(Worker);
                ths[i].Name = "Thread_" + i;
                ths[i].Start();
                Console.WriteLine(ths[i].Name + " Starting ...");
            }

            for (int i = 0; i < Environment.ProcessorCount; ++i)
            {
                ths[i].Join();

            }
            Console.WriteLine("Finished !");
        }

        private void Worker()
        {
            string Img_name;
            string th_name = Thread.CurrentThread.Name;

            while (path_Imgs.TryDequeue(out Img_name))
            {
                if (wait.WaitOne(0))
                    break;
                Console.WriteLine(th_name + ": " + model.PredictModel(OnnxGeneral.ImageModel(Img_name)));

            }

            if (stop == 0)
                Console.WriteLine(th_name + " Stop normally !");
            else
                Console.WriteLine(th_name + " Stop by Ctrl+C !");
        }

        protected static void myHandler(object sender, ConsoleCancelEventArgs args)
        {
            stop = 1;
            wait.Set();
            Console.WriteLine("Stop threads by Ctrl+C ...");
            args.Cancel = true;
        }
    }
}
