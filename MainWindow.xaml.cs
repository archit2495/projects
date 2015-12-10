
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Drawing;
using System.Windows.Forms;

namespace WpfApplication3
{

    /// <summary>
    /// interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        static int NUM_DATA = 7291;    //used for training
        int INT_MIN = -1000000;
        static int NUM_DATA1 = 1000;  //used for testing
        long q;
        int LO = 0;
        int HI = 1;
        int X = 16;
        int Y = 16;
        static int N = 256; //total number of pixels
        static int M = 10;  //total possible outcomes
        int BIAS = 1;
        int flag = 0;
        int[] hsh = new int[M]; //matrix to store results of training
        double[,] Input = new double[NUM_DATA,N]; //input matrix for training
        double[,] Input1=new double[NUM_DATA1, N]; //input matrix for testing on test files
        double[,] Input2 = new double[1,N]; //input matrix for testing on custom image
        int[,] Output = new int[NUM_DATA,M];  //output matrix for training
        int[,] Output1=new int[NUM_DATA1,M];   //output matrix for testing
        int[,] Output2 = new int[1, M]; //output matrix for testing on custom image
       
        static NET Net;
        public struct LAYER{                     /* A LAYER OF A NET:                     */
        public int Units;         /* - number of units in this layer       */
        public double[] Activation;    /* - activation of ith unit              */
        public double[] Output;        /* - output of ith unit                  */
        public double[] Error;         /* - error term of ith unit              */
        public double[,] Weight;        /* - connection weights to ith unit      */
        };

        public struct NET{                     /* A NET:                                */
        public LAYER InputLayer;    /* - input layer                         */
        public LAYER OutputLayer;   /* - output layer                        */
        public double Eta;           /* - learning rate                       */
        public double Error;         /* - total net error                     */
        public double Epsilon;       /* - net error to terminate training     */
        };


        
         public void GenerateNetwork()
        {

            Net.InputLayer = new LAYER();       
            Net.OutputLayer = new LAYER();

            Net.InputLayer.Units = N;
            Net.InputLayer.Output = new double[N + 1];      /* Allocating Space */
            Net.InputLayer.Output[0] = BIAS;

            Net.OutputLayer.Units = M;
            Net.OutputLayer.Activation=new double[M+1];
            Net.OutputLayer.Output = new double[M + 1];
            Net.OutputLayer.Error = new double[M + 1];
            Net.OutputLayer.Weight = new double[M + 1,N+1];

            Net.Eta = 0.001;
            Net.Epsilon =1.25;
        }
       


        void SetInput(double[,] Input,int index)
        {
            int i;

            for (i = 1; i <= Net.InputLayer.Units; i++)
            {
                Net.InputLayer.Output[i] = Input[index,i - 1];
            }
        }

      

        public void PropagateNet()
        {
            int i, j, index=0;
            double Sum;
            double Max =INT_MIN;
            for (i = 1; i <= Net.OutputLayer.Units; i++)
            {
                Sum = 0;
                for (j = 0; j <= Net.InputLayer.Units; j++)
                {
                    Sum += ((Net.OutputLayer.Weight[i,j]) * (Net.InputLayer.Output[j]));
                }
                Net.OutputLayer.Activation[i] = Sum;
                Net.OutputLayer.Output[i] = LO;

                if (Sum > Max)
                {
                    Max = Sum;
                    index = i;
                }
               
            }
            Net.OutputLayer.Output[index] = HI;
        }


        /******************************************************************************
                             P R O P A G A T I N G   S I G N A L S
         ******************************************************************************/

       

        /******************************************************************************
                               A D J U S T I N G   W E I G H T S
         ******************************************************************************/

        void ComputeOutputError(int[,] Target,int index)
        {
            int i;
            double Err;

            Net.Error = 0;
            for (i = 1; i <= Net.OutputLayer.Units; i++)
            {
                Err = Target[index,i - 1] - Net.OutputLayer.Activation[i];
                Net.OutputLayer.Error[i] = Err;
                Net.Error += 0.5 * (Err*Err);
            }
        }

        void AdjustWeights()
        {
            int i, j;
            double Out;
            double Err;

            for (i = 1; i <= Net.OutputLayer.Units; i++)
            {
                for (j = 0; j <= Net.InputLayer.Units; j++)
                {
                    Out = Net.InputLayer.Output[j];
                    Err = Net.OutputLayer.Error[i];
                    Net.OutputLayer.Weight[i,j] += Net.Eta * Err * Out;
                }
            }
        }

        /******************************************************************************
                              S I M U L A T I N G   T H E   N E T
         ******************************************************************************/

        void SimulateNet(double[,] Input, int[,] Target, bool Training, int index)
        {

            SetInput(Input, index);
            PropagateNet();

            ComputeOutputError( Target,index);
            if (Training)
                AdjustWeights();
        }

        // Function to Train our Algorithm

        public void train()
        {

            int k;
            if(flag==1)
            {
                for (k = 0; k < M; k++)
                {
                    trainBox.Text += "\t" + k + "\t" + hsh[k] + "\n";
                }
                return;
            }
            else
            {
                flag = 1;
            }
            int i, dig,n,m;
            double x,Error;
            bool Stop;
            Net.Error = 0;
            ////////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////// Constructing Input Matrix ////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////
            for(i=0;i<M;i++)
            {
                hsh[i] = 0;
            }
            string file_name = "C:\\Users\\DELL LAPTOP\\Downloads\\Train1.txt"; //path of train file


            if (System.IO.File.Exists(file_name) == true)
            {
                string text = System.IO.File.ReadAllText(file_name);

                string[] temp = text.Split(' ');

                dig = 0;
                
                for (i = 0; i < temp.Length; i++)
                {
                    if (i/257==NUM_DATA)
                    {
                        break;
                    }
                    x = double.Parse(temp[i]);

                    if (i % 257 == 0)
                    {
                        dig = (int)x;
                        hsh[dig]++;
                        Output[i / 257,dig] = HI;
                    }
                    else
                    {
                        dig = (int)x;
                        Input[i / 257,(i % 257) - 1] = x;
                    }


                }
                

                trainBox.Text += "\tDigit\tSample Size\n";
                trainBox.Text += "\t-----\t-----------\n";

                for (i = 0; i < M; i++)
                {
                    trainBox.Text += "\t" + i + "\t" + hsh[i] + "\n";
                }


            }
            else
            {
                System.Windows.MessageBox.Show("No Such File " + file_name);
            }

            do
            {
                Error = 0;
                Stop = true;
                for (n = 0; n < NUM_DATA; n++)
                {
                    SimulateNet(Input, Output,true,n);
                    Error = Math.Max(Error, Net.Error);
                    Stop = Stop &&(Net.Error < Net.Epsilon);
                }
               
            } while (!Stop);
          
        }

        // Function to test our Algorithm

        public void test()
        {
            double correct = 0,x;
            int total = 0;
            int i, dig, n, m;
            
            ////////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////// Constructing Input Matrix ////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////
           
            string file_name = "C:\\Users\\DELL LAPTOP\\Downloads\\Test.txt"; //path of test file

            trainBox.Text = null;

            if (System.IO.File.Exists(file_name) == true)
            {
                string text = System.IO.File.ReadAllText(file_name);

                string[] temp = text.Split(' ','\n');

                dig = 0;
                for (i = 0; i < temp.Length; i++)
                {

                    if(i/257==NUM_DATA1)
                    {
                        break;
                    }
                    x = double.Parse(temp[i]);
                  
                        if (i % 257 == 0)
                        {
                            dig = (int)x;

                            Output1[i / 257, dig] = HI;
                        }
                        else
                        {
                            Input1[i / 257, (i % 257) - 1] = x;
                        }

                    
                }

                
                

            }
            else
            {
                System.Windows.MessageBox.Show("No Such File " + file_name);
            }

            for (n = 0; n < NUM_DATA1; n++)
            {                                                     //Here we are testing
                SimulateNet(Input1, Output1,false, n);
                int count=0, Index=0,j;
                for (j = 0; j < M; j++)
                {
                    if (Net.OutputLayer.Output[j + 1]==HI)
                    {
                        Index = j;
                        count++;
                    }
                }
                if (count>0)
                {
                    if (Output1[total,Index] == HI)
                    {
                        correct++;
                    }
                    total++;
                }
               
            }

            double num = (double)NUM_DATA1;
            double efficiency = correct / num;
            System.Windows.MessageBox.Show("Efficiency " + (efficiency) *100+"%");

            
        }


        // To Convert Image to Vector of Size 256

        public void Imageproc()
        {

            ////////////////////////////////////////////////////////////////////////////////////////////////////
            ///////////////////////////////////// Image Processing ///////////////////////////////////////////// 
            ////////////////////////////////////////////////////////////////////////////////////////////////////

            int i, j, k, r1, g1, b1;
            double x;


            string file_name;

            OpenFileDialog open = new OpenFileDialog();

            open.InitialDirectory = "C:\\Users\\DELL LAPTOP\\Desktop\\Machine_learning";
            open.Filter = "JPEG Files (*.jpg)|*.jpg|PNG Files (*.png)|*.png|BMP Files (*.bmp)|*.bmp|All files (*.*)|*.*";

            open.ShowDialog();

            file_name = open.FileName;
            
            
            BitmapImage img = new BitmapImage();
            img.BeginInit();
            img.UriSource = new Uri(open.FileName);
            img.EndInit();
            image.Source = img;
            

            Bitmap b = new Bitmap(file_name);
            Bitmap a = new Bitmap(b, 16, 16);
            
            k = 0;

            for (i = 0; i < a.Height; i++)
            {
                for (j = 0; j < a.Width; j++)
                {
                    System.Drawing.Color c = a.GetPixel(j, i);
                    r1 = c.R;
                    g1 = c.G;
                    b1 = c.B;

                    x = r1 + g1 + b1;
                    x = x / 3.0;
                    
                    x = x / 127.5;
                    x = -(x - 1);

                    Input2[0,k] = x;
                    k++;

                }
            }

        }

        // To check Individual Image

      
         public MainWindow()
        {
           // InitializeComponent();
            GenerateNetwork();
        }

        private void trainButton_Click(object sender, RoutedEventArgs e)
        {
            trainBox.Text = null;
            image.Source = null;
            trainBox.Text += "Training....\n";
            DateTime baseTime = new DateTime(1970, 1, 1, 0, 0, 0);
            DateTime nowInUTC = DateTime.UtcNow;
            q = (nowInUTC - baseTime).Ticks;
            Console.WriteLine(q);
            // Function to Calculate Size of Data Set for each Digit
            train();               // Function to Train our Algorithm

        }

        private void testButton_Click(object sender, RoutedEventArgs e)
        {
            
            trainBox.Text = null;
            
            trainBox.Text += "Testing....\n";
           
            test();                // Function to test our Algorithm

        }

        private void doubleButton_Click(object sender, RoutedEventArgs e)
        {
                    // To check Individual Image
            Imageproc();
            SimulateNet(Input2, Output2, false,0);
            int  j;
            for (j = 0; j < M; j++)
            {
                if (Net.OutputLayer.Output[j + 1] == HI)
                {
                   
                    trainBox.Text = null;
                    trainBox.Text += ("Digit = " + j);
                }
            }
        }

        private void closeButton_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }
        
        
    }
}
