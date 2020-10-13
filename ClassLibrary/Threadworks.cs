using System;
using System.IO;
using System.Collections.Concurrent;
using System.Threading;

namespace ClassLibrary
{
    public class Threadworks
    {
        private static AutoResetEvent wait_for_write = new AutoResetEvent(true);
        
        public static CancellationTokenSource cancelTokenSource = new CancellationTokenSource();
        public static CancellationToken token = cancelTokenSource.Token;
        
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
                path_Imgs = new ConcurrentQueue<string>(Directory.GetFiles(_path, "*.jpeg"));
            }
            catch (DirectoryNotFoundException)
            {
                Console.WriteLine("These files don't exist!");
                return;
            }
            path_Imgs = new ConcurrentQueue<string>(Directory.GetFiles(_path, "*.jpeg"));
            
            Console.WriteLine("Press Enter to stop threads ...");

            var Wait_for_enter = new Thread(new ThreadStart(() =>
            {
                while (Console.ReadKey().Key != ConsoleKey.Enter) { }
                Threadworks.cancelTokenSource.Cancel();
            }));
            Wait_for_enter.Start();                        
            
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
                if (token.IsCancellationRequested)
                {
                    stop++;
                    break;
                }
                
                _result = th_name + ": " + model.PredictModel(OnnxGeneral.ImageModel(Img_name)) + ", file path: " + Img_name;
                wait_for_write.WaitOne();
                results.Add(_result);
                wait_for_write.Set();
            }
            if (stop == 0)
                Console.WriteLine(th_name + " Stoped normally !");
            else
                Console.WriteLine(th_name + " Stoped by Enter !");
        }

        public override string ToString()
        {
            for(int i = 0; i < results.Count; i++)
            {
                Console.WriteLine(results[i]);
            }
            
            return "\n";
        }
        
    }
}
