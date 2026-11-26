with Ada.Text_IO;
with SPARK.Containers.Types; use SPARK.Containers.Types;
with Inst;
use Inst.Int_Lists;

procedure Test with SPARK_Mode is

   procedure Assert (B : Boolean; S : String)
   with Pre => B;
   procedure Assert (B : Boolean; S : String) is
   begin
      if not B then
         Ada.Text_IO.Put_Line (S);
      end if;
   end Assert;

   procedure Test_Clear
   with Pre => True;

   procedure Test_Clear is separate;

begin
   Test_Clear;
end Test;
