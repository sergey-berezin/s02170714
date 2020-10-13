using System;
using System.IO;
using System.Collections.Concurrent;
using System.Threading;

namespace ClassLibrary
{
    public class Threadworks
    {
        private static ManualResetEvent wait = new ManualResetEvent(false);
        private static AutoResetEvent wait_for_write = new AutoResetEvent(false);
        private ConcurrentQueue<string> path_Imgs;

        private OnnxGeneral model;
        private string _path;
        public static int stop = 0;
        private List<string> results = new List<string>();
        
        public Threadworks(string path, OnnxGeneral onnxModel)
        {
            model = onnxModel;
            _path = path;
        }

        public void Run()
        {
            
            try
            {
                path_Imgs = new ConcurrentQueue<string>(Directory.GetFiles(_path, "*.jpg"));
            }
            catch (DirectoryNotFoundException)
            {
                Console.WriteLine("These files don't exist!");
                return;
            }
            path_Imgs = new ConcurrentQueue<string>(Directory.GetFiles(_path, "*.jpeg"));
            
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
            string _result;
            int time = 0;
            
            while (path_Imgs.TryDequeue(out Img_name))
            {
                if (wait.WaitOne(0))
                    break;
                //Console.WriteLine(th_name + ": " + model.PredictModel(OnnxGeneral.ImageModel(Img_name)) + ", file path: " + Img_name);

                _result = th_name + ": " + model.PredictModel(OnnxGeneral.ImageModel(Img_name)) + ", file path: " + Img_name;                
                if(th_name == "Thread_0" && time == 0)
                {
                    results.Add(_result);
                    Thread.Sleep(500);
                    time++;
                    wait_for_write.Set();
                }
                else
                {
                    wait_for_write.WaitOne();
                    results.Add(_result);
                    wait_for_write.Set();
                }
                
            }
            if (stop == 0)
                Console.WriteLine(th_name + " Stop normally !");
            else
                Console.WriteLine(th_name + " Stop by Ctrl+C !");
        }

        public override string ToString()
        {
            for(int i = 0; i < results.Count; i++)
            {
                Console.WriteLine(results[i]);
            }
            
            return "\n";
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
