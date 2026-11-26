with SPARK.Containers.Types; use SPARK.Containers.Types;
with Inst;                   use Inst.Int_Vectors;

package Test_Resize with SPARK_Mode is

   --  Unbounded containers are resized automatically when they grow. Test
   --  the capability by inserting enough elements in a container.

   function Test_Resize_Big return Vector with
     Post => Length (Test_Resize_Big'Result) = 1000;

end Test_Resize;
