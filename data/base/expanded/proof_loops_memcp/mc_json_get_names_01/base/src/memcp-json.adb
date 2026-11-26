--  Memcp.Json body: the json crate is instantiated here and nowhere else, and
--  every value it yields is read through an observer that never leaves this
--  body.

with Ada.Containers;
use type Ada.Containers.Count_Type;

package body Memcp.Json
  with SPARK_Mode => On
is

   use type Types.Value_Kind;

   ---------------
   -- Get_Names --
   ---------------

   function Get_Names
     (M : access constant Types.JSON_Value) return Memcp.Store.Name_List
   is separate;

end Memcp.Json;
