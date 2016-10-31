using System;
namespace LinearRegression
{
  class LinearRegressionClass
  {
    static void Main(string[] args)
    {

      LinearRegressionClass m_LinearRegressionClass = new LinearRegressionClass();
      m_LinearRegressionClass.LinearRegression();



    } // Main


    public void LinearRegression()
    {
      Console.WriteLine("\nBegin linear regression demo\n");
      int rows = 100;
      int seed = 1;

      Console.WriteLine("Creating " + rows + " rows synthetic data");
      double[][] data = DummyData(rows, seed);
      Console.WriteLine("Done\n");

      //double[][] data = MatrixLoad("..\\..\\LLuviaData.txt", true, ',');

      Console.WriteLine("temp-humidity-press-rain data:\n");
      ShowMatrix(data, 2);

      Console.WriteLine("\nCreating design matrix from data");
      double[][] design = Design(data); // 'design matrix'
      Console.WriteLine("Done\n");

      Console.WriteLine("Design matrix:\n");
      ShowMatrix(design, 2);

      Console.WriteLine("\nFinding coefficients using inversion");
      double[] coef = Solve(design); // use design matrix
      Console.WriteLine("Done\n");

      Console.WriteLine("Coefficients are:\n");
      ShowVector(coef, 4);
      Console.WriteLine("");

      Console.WriteLine("Computing R-squared\n");
      double R2 = RSquared(data, coef); // use initial data
      Console.WriteLine("R-squared = " + R2.ToString("F4"));

      Console.WriteLine("\nPredicting LLuvia for ");
      Console.WriteLine("Temp = -5");
      Console.WriteLine("humidity = 310");
      Console.WriteLine("press = 1003");

      double y = LLuvia(-5, 310, 1003, coef);
      Console.WriteLine("\nPredicted LLuvia = " + y.ToString("F2"));

      Console.WriteLine("\nEnd linear regression demo\n");
      Console.ReadLine();


    } // Main


    // private double[][] Bioma(int biomaCode)
    // {
    //   double[][] retorno = new double[4][4];
    //   switch (biomaCode)
    //   {
    // case 1:
    //     Console.WriteLine("Tundra");
    //     retorno[0][0] = -15; //temp min
    //     retorno[0][1] = -5; //temp max
    //
    //     retorno[1][0] = 300; //precip min
    //     retorno[1][1] = 350; //precip max
    //
    //     retorno[1][0] = 300; //precip min
    //     retorno[1][1] = 350; //precip max
    //     break;
    // case 2:
    //     Console.WriteLine("Bosque caducifolio");
    //     break;
    // default:
    //     Console.WriteLine("Default case");
    //     break;
    //   }
    //   return retorno;
    //
    // }


    private double LLuvia(double x1, double x2, double x3, double[] coef)
    {
      // x1 = education, x2 = work, x3 = sex
      double result; // the constant
      result = coef[0] + (x1 * coef[1]) + (x2 * coef[2]) + (x3 * coef[3]);
      return result;
    }

    private double RSquared(double[][] data, double[] coef)
    {
      // 'coefficient of determination'
      int rows = data.Length;
      int cols = data[0].Length;

      // 1. compute mean of y
      double ySum = 0.0;
      for (int i = 0; i < rows; ++i)
        ySum += data[i][cols - 1]; // last column
      double yMean = ySum / rows;

      // 2. sum of squared residuals & tot sum squares
      double ssr = 0.0;
      double sst = 0.0;
      double y; // actual y value
      double predictedY; // using the coef[]
      for (int i = 0; i < rows; ++i)
      {
        y = data[i][cols - 1]; // get actual y

        predictedY = coef[0]; // start w/ intercept constant
        for (int j = 0; j < cols - 1; ++j) // j is col of data
          predictedY += coef[j+1] * data[i][j]; // careful

        ssr += (y - predictedY) * (y - predictedY);
        sst += (y - yMean) * (y - yMean);
      }

      if (sst == 0.0)
        throw new Exception("All y values equal");
      else
        return 1.0 - (ssr / sst);
     }

    private double[][] DummyData(int rows, int seed)
    {
      // generate dummy data for linear regression problem
      // double b0 = 15.0; // y
      // double b1 = -0.8; // temp centrigrados
      // double b2 = 0.5; // humidity mm
      // double b3 = 3.1; // press =mmg
      Random rnd = new Random(seed);

      double[][] result = new double[rows][];
      for (int i = 0; i < rows; ++i)
        result[i] = new double[4];

      for (int i = 0; i < rows; ++i)
      {
        int temp = rnd.Next(-15, 6); // -12, 4]
        int hum = rnd.Next(300, 351); // [300, 350]
        int press = rnd.Next(1000, 1005); // 1000 or 1004
        double y = rnd.Next(400, 703);
        // double y = b0 + (b1 * temp) + (b2 * hum) + (b3 * press);
        // y += 10.0 * rnd.NextDouble() - 5.0; // random [-5 +5]

        result[i][0] = temp;
        result[i][1] = hum;
        result[i][2] = press;
        result[i][3] = y; // cant precipitacion
      }
      return result;
    }

    private double[][] Design(double[][] data)
    {
      // add a leading col of 1.0 values
      int rows = data.Length;
      int cols = data[0].Length;
      double[][] result = MatrixCreate(rows, cols + 1);
      for (int i = 0; i < rows; ++i)
        result[i][0] = 1.0;

      for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
          result[i][j + 1] = data[i][j];

      return result;
    }

    private double[] Solve(double[][] design)
    {
      // find linear regression coefficients
      // 1. peel off X matrix and Y vector
      int rows = design.Length;
      int cols = design[0].Length;
      double[][] X = MatrixCreate(rows, cols - 1);
      double[][] Y = MatrixCreate(rows, 1); // a column vector

      int j;
      for (int i = 0; i < rows; ++i)
      {
        for (j = 0; j < cols - 1; ++j)
        {
          X[i][j] = design[i][j];
        }
        Y[i][0] = design[i][j]; // last column
      }

      // 2. B = inv(Xt * X) * Xt * y
      double[][] Xt = MatrixTranspose(X);
      double[][] XtX = MatrixProduct(Xt, X);
      double[][] inv = MatrixInverse(XtX);
      double[][] invXt = MatrixProduct(inv, Xt);

      double[][] mResult = MatrixProduct(invXt, Y);
      double[] result = MatrixToVector(mResult);
      return result;
    } // Solve


    private void ShowMatrix(double[][] m, int dec)
    {
      for (int i = 0; i < m.Length; ++i)
      {
        for (int j = 0; j < m[i].Length; ++j)
        {
          Console.Write(m[i][j].ToString("F" + dec) + "  ");
        }
        Console.WriteLine("");
      }
    }

    private void ShowVector(double[] v, int dec)
    {
      for (int i = 0; i < v.Length; ++i)
        Console.Write(v[i].ToString("F" + dec) + "  ");
      Console.WriteLine("");
    }


    // ===== Matrix routines

    private double[][] MatrixCreate(int rows, int cols)
    {
      // allocates/creates a matrix initialized to all 0.0
      // do error checking here
      double[][] result = new double[rows][];
      for (int i = 0; i < rows; ++i)
        result[i] = new double[cols];
      return result;
    }

    // -------------------------------------------------------------

    private double[][] MatrixRandom(int rows, int cols,
      double minVal, double maxVal, int seed)
    {
      // return a matrix with random values
      Random ran = new Random(seed);
      double[][] result = MatrixCreate(rows, cols);
      for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
          result[i][j] = (maxVal - minVal) *
            ran.NextDouble() + minVal;
      return result;
    }

    // -------------------------------------------------------------

    private double[][] MatrixLoad(string file, bool header,
      char sep)
    {
      // load a matrix from a text file
      string line = "";
      string[] tokens = null;
      int ct = 0;
      int rows, cols;
      // determined # rows and cols
      System.IO.FileStream ifs =
        new System.IO.FileStream(file, System.IO.FileMode.Open);
      System.IO.StreamReader sr =
        new System.IO.StreamReader(ifs);
      while ((line = sr.ReadLine()) != null)
      {
        ++ct;
        tokens = line.Split(sep); // do validation here
      }
      sr.Close(); ifs.Close();
      if (header == true)
        rows = ct - 1;
      else
        rows = ct;
      cols = tokens.Length;
      double[][] result = MatrixCreate(rows, cols);

      // load
      int i = 0; // row index
      ifs = new System.IO.FileStream(file, System.IO.FileMode.Open);
      sr = new System.IO.StreamReader(ifs);

      if (header == true)
        line = sr.ReadLine();  // consume header
      while ((line = sr.ReadLine()) != null)
      {
        tokens = line.Split(sep);
        for (int j = 0; j < cols; ++j)
          result[i][j] = double.Parse(tokens[j]);
        ++i; // next row
      }
      sr.Close(); ifs.Close();
      return result;
    }

    // -------------------------------------------------------------

    private double[] MatrixToVector(double[][] matrix)
    {
      // single column matrix to vector
      int rows = matrix.Length;
      int cols = matrix[0].Length;
      if (cols != 1)
        throw new Exception("Bad matrix");
      double[] result = new double[rows];
      for (int i = 0; i < rows; ++i)
        result[i] = matrix[i][0];
      return result;
    }

    // -------------------------------------------------------------

    private double[][] MatrixIdentity(int n)
    {
      // return an n x n Identity matrix
      double[][] result = MatrixCreate(n, n);
      for (int i = 0; i < n; ++i)
        result[i][i] = 1.0;

      return result;
    }

    // -------------------------------------------------------------

    private string MatrixAsString(double[][] matrix, int dec)
    {
      string s = "";
      for (int i = 0; i < matrix.Length; ++i)
      {
        for (int j = 0; j < matrix[i].Length; ++j)
          s += matrix[i][j].ToString("F"+dec).PadLeft(8) + " ";
        s += Environment.NewLine;
      }
      return s;
    }

    // -------------------------------------------------------------

    private bool MatrixAreEqual(double[][] matrixA,
      double[][] matrixB, double epsilon)
    {
      // true if all values in matrixA == corresponding values in matrixB
      int aRows = matrixA.Length; int aCols = matrixA[0].Length;
      int bRows = matrixB.Length; int bCols = matrixB[0].Length;
      if (aRows != bRows || aCols != bCols)
        throw new Exception("Non-conformable matrices in MatrixAreEqual");

      for (int i = 0; i < aRows; ++i) // each row of A and B
        for (int j = 0; j < aCols; ++j) // each col of A and B
          //if (matrixA[i][j] != matrixB[i][j])
          if (Math.Abs(matrixA[i][j] - matrixB[i][j]) > epsilon)
            return false;
      return true;
    }

    // -------------------------------------------------------------

    private double[][] MatrixProduct(double[][] matrixA, double[][] matrixB)
    {
      int aRows = matrixA.Length; int aCols = matrixA[0].Length;
      int bRows = matrixB.Length; int bCols = matrixB[0].Length;
      if (aCols != bRows)
        throw new Exception("Non-conformable matrices in MatrixProduct");

      double[][] result = MatrixCreate(aRows, bCols);

      for (int i = 0; i < aRows; ++i) // each row of A
        for (int j = 0; j < bCols; ++j) // each col of B
          for (int k = 0; k < aCols; ++k) // could use k < bRows
            result[i][j] += matrixA[i][k] * matrixB[k][j];

      //Parallel.For(0, aRows, i =>
      //  {
      //    for (int j = 0; j < bCols; ++j) // each col of B
      //      for (int k = 0; k < aCols; ++k) // could use k < bRows
      //        result[i][j] += matrixA[i][k] * matrixB[k][j];
      //  }
      //);

      return result;
    }

    // -------------------------------------------------------------

    private double[] MatrixVectorProduct(double[][] matrix, double[] vector)
    {
      // result of multiplying an n x m matrix by a m x 1 column vector (yielding an n x 1 column vector)
      int mRows = matrix.Length; int mCols = matrix[0].Length;
      int vRows = vector.Length;
      if (mCols != vRows)
        throw new Exception("Non-conformable matrix and vector in MatrixVectorProduct");
      double[] result = new double[mRows]; // an n x m matrix times a m x 1 column vector is a n x 1 column vector
      for (int i = 0; i < mRows; ++i)
        for (int j = 0; j < mCols; ++j)
          result[i] += matrix[i][j] * vector[j];
      return result;
    }

    // -------------------------------------------------------------

    private double[][] MatrixDecompose(double[][] matrix, out int[] perm,
      out int toggle)
    {
      // Doolittle LUP decomposition with partial pivoting.
      // returns: result is L (with 1s on diagonal) and U;
      // perm holds row permutations; toggle is +1 or -1 (even or odd)
      int rows = matrix.Length;
      int cols = matrix[0].Length;
      if (rows != cols)
        throw new Exception("Non-square mattrix");

      int n = rows; // convenience

      double[][] result = MatrixDuplicate(matrix); //

      perm = new int[n]; // set up row permutation result
      for (int i = 0; i < n; ++i) { perm[i] = i; }

      toggle = 1; // toggle tracks row swaps

      for (int j = 0; j < n - 1; ++j) // each column
      {
        double colMax = Math.Abs(result[j][j]);
        int pRow = j;
        //for (int i = j + 1; i < n; ++i) // deprecated
        //{
        //  if (result[i][j] > colMax)
        //  {
        //    colMax = result[i][j];
        //    pRow = i;
        //  }
        //}

        for (int i = j + 1; i < n; ++i) // reader Matt V needed this:
        {
          if (Math.Abs(result[i][j]) > colMax)
          {
            colMax = Math.Abs(result[i][j]);
            pRow = i;
          }
        }
        // Not sure if this approach is needed always, or not.

        if (pRow != j) // if largest value not on pivot, swap rows
        {
          double[] rowPtr = result[pRow];
          result[pRow] = result[j];
          result[j] = rowPtr;

          int tmp = perm[pRow]; // and swap perm info
          perm[pRow] = perm[j];
          perm[j] = tmp;

          toggle = -toggle; // adjust the row-swap toggle
        }

        // -------------------------------------------------------------
        // This part added later (not in original code)
        // and replaces the 'return null' below.
        // if there is a 0 on the diagonal, find a good row
        // from i = j+1 down that doesn't have
        // a 0 in column j, and swap that good row with row j

        if (result[j][j] == 0.0)
        {
          // find a good row to swap
          int goodRow = -1;
          for (int row = j + 1; row < n; ++row)
          {
            if (result[row][j] != 0.0)
              goodRow = row;
          }

          if (goodRow == -1)
            throw new Exception("Cannot use Doolittle's method");

          // swap rows so 0.0 no longer on diagonal
          double[] rowPtr = result[goodRow];
          result[goodRow] = result[j];
          result[j] = rowPtr;

          int tmp = perm[goodRow]; // and swap perm info
          perm[goodRow] = perm[j];
          perm[j] = tmp;

          toggle = -toggle; // adjust the row-swap toggle
        }
        // -------------------------------------------------------------

        //if (Math.Abs(result[j][j]) < 1.0E-20) // deprecated
        //  return null; // consider a throw

        for (int i = j + 1; i < n; ++i)
        {
          result[i][j] /= result[j][j];
          for (int k = j + 1; k < n; ++k)
          {
            result[i][k] -= result[i][j] * result[j][k];
          }
        }

      } // main j column loop

      return result;
    } // MatrixDecompose

    // -------------------------------------------------------------

    private double[][] MatrixInverse(double[][] matrix)
    {
      int n = matrix.Length;
      double[][] result = MatrixDuplicate(matrix);

      int[] perm;
      int toggle;
      double[][] lum = MatrixDecompose(matrix, out perm, out toggle);
      if (lum == null)
        throw new Exception("Unable to compute inverse");

      double[] b = new double[n];
      for (int i = 0; i < n; ++i)
      {
        for (int j = 0; j < n; ++j)
        {
          if (i == perm[j])
            b[j] = 1.0;
          else
            b[j] = 0.0;
        }

        double[] x = HelperSolve(lum, b); // use decomposition

        for (int j = 0; j < n; ++j)
          result[j][i] = x[j];
      }
      return result;
    }

    // -------------------------------------------------------------

    private double[][] MatrixTranspose(double[][] matrix)
    {
      int rows = matrix.Length;
      int cols = matrix[0].Length;
      double[][] result = MatrixCreate(cols, rows); // note indexing
      for (int i = 0; i < rows; ++i)
      {
        for (int j = 0; j < cols; ++j)
        {
          result[j][i] = matrix[i][j];
        }
      }
      return result;
    } // TransposeMatrix

    // -------------------------------------------------------------

    private double MatrixDeterminant(double[][] matrix)
    {
      int[] perm;
      int toggle;
      double[][] lum = MatrixDecompose(matrix, out perm, out toggle);
      if (lum == null)
        throw new Exception("Unable to compute MatrixDeterminant");
      double result = toggle;
      for (int i = 0; i < lum.Length; ++i)
        result *= lum[i][i];
      return result;
    }

    // -------------------------------------------------------------

    private double[] HelperSolve(double[][] luMatrix, double[] b)
    {
      // before calling this helper, permute b using the perm array
      // from MatrixDecompose that generated luMatrix
      int n = luMatrix.Length;
      double[] x = new double[n];
      b.CopyTo(x, 0);

      for (int i = 1; i < n; ++i)
      {
        double sum = x[i];
        for (int j = 0; j < i; ++j)
          sum -= luMatrix[i][j] * x[j];
        x[i] = sum;
      }

      x[n - 1] /= luMatrix[n - 1][n - 1];
      for (int i = n - 2; i >= 0; --i)
      {
        double sum = x[i];
        for (int j = i + 1; j < n; ++j)
          sum -= luMatrix[i][j] * x[j];
        x[i] = sum / luMatrix[i][i];
      }

      return x;
    }

    // -------------------------------------------------------------

    //private double[] SystemSolve(double[][] A, double[] b)
    //{
    //  // Solve Ax = b
    //  int n = A.Length;

    //  // 1. decompose A
    //  int[] perm;
    //  int toggle;
    //  double[][] luMatrix = MatrixDecompose(A, out perm, out toggle);
    //  if (luMatrix == null)
    //    return null;

    //  // 2. permute b according to perm[] into bp
    //  double[] bp = new double[b.Length];
    //  for (int i = 0; i < n; ++i)
    //    bp[i] = b[perm[i]];

    //  // 3. call helper
    //  double[] x = HelperSolve(luMatrix, bp);
    //  return x;
    //} // SystemSolve

    // -------------------------------------------------------------

    private double[][] MatrixDuplicate(double[][] matrix)
    {
      // allocates/creates a duplicate of a matrix
      double[][] result = MatrixCreate(matrix.Length, matrix[0].Length);
      for (int i = 0; i < matrix.Length; ++i) // copy the values
        for (int j = 0; j < matrix[i].Length; ++j)
          result[i][j] = matrix[i][j];
      return result;
    }

    // -------------------------------------------------------------

    private double[][] ExtractLower(double[][] matrix)
    {
      // lower part of a Doolittle decomp (1.0s on diagonal, 0.0s in upper)
      int rows = matrix.Length; int cols = matrix[0].Length;
      double[][] result = MatrixCreate(rows, cols);
      for (int i = 0; i < rows; ++i)
      {
        for (int j = 0; j < cols; ++j)
        {
          if (i == j)
            result[i][j] = 1.0;
          else if (i > j)
            result[i][j] = matrix[i][j];
        }
      }
      return result;
    }

    private double[][] ExtractUpper(double[][] matrix)
    {
      // upper part of a Doolittle decomp (0.0s in the strictly lower part)
      int rows = matrix.Length; int cols = matrix[0].Length;
      double[][] result = MatrixCreate(rows, cols);
      for (int i = 0; i < rows; ++i)
      {
        for (int j = 0; j < cols; ++j)
        {
          if (i <= j)
            result[i][j] = matrix[i][j];
        }
      }
      return result;
    }

    // -------------------------------------------------------------

    private double[][] PermArrayToMatrix(int[] perm)
    {
      // convert Doolittle perm array to corresponding perm matrix
      int n = perm.Length;
      double[][] result = MatrixCreate(n, n);
      for (int i = 0; i < n; ++i)
        result[i][perm[i]] = 1.0;
      return result;
    }

    private double[][] UnPermute(double[][] luProduct, int[] perm)
    {
      // unpermute product of Doolittle lower * upper matrix according to perm[]
      // no real use except to demo LU decomposition, or for consistency testing
      double[][] result = MatrixDuplicate(luProduct);

      int[] unperm = new int[perm.Length];
      for (int i = 0; i < perm.Length; ++i)
        unperm[perm[i]] = i;

      for (int r = 0; r < luProduct.Length; ++r)
        result[r] = luProduct[unperm[r]];

      return result;
    } // UnPermute


    // =====

  } // Program

} // ns
