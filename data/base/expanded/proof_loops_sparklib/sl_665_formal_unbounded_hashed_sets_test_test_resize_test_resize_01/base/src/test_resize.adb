with SPARK.Containers.Types; use SPARK.Containers.Types;
with Inst_Gen;

--  Unbounded containers are resized automatically when they grow. Test
--  the capability by inserting enough elements in a container.
--  Disable execution of contracts on this test as their prohibitively slow.

procedure Test_Resize with SPARK_Mode is
   pragma Assertion_Policy (SPARKlib_Full => Ignore);
   package Inst is new Inst_Gen (100);
   use Inst.Int_Sets;
   X : Set;

   procedure Insert_Elements
   with Pre => Is_Empty (X);

   procedure Insert_Elements is separate;

begin
   Insert_Elements;
end Test_Resize;
