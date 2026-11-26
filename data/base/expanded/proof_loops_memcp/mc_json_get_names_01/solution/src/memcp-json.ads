--  Memcp.Json: the tool layer's JSON marshalling helper. Inbound, it parses an
--  `arguments` object -- raw JSON text off the seam -- into typed fields;
--  outbound, it renders scalars as JSON text. No JSON access type crosses this
--  package's boundary: every getter hands back a plain value.

with Json_Types;

with Memcp.Store;

package Memcp.Json
  with SPARK_Mode => On
is

   package Types renames Json_Types;
   --  Upstream this is a body-local instantiation of JSON.Types, and no json
   --  access type reaches this spec. The isolation takes the member observer
   --  as a parameter instead of looking it up in a Doc, so the name has to be
   --  visible here.

   function Get_Names
     (M : access constant Types.JSON_Value) return Memcp.Store.Name_List;
   --  The string-array member M as a Memcp.Store.Name_List. Null or
   --  not-an-array yields an empty list, and non-string elements are skipped.
   --  Uncapped, because the Store refuses a list longer than Max_Filter_Terms.
   --  @param M An observer of the member, or null when it is absent.
   --  @return The array's string elements as a Name_List (empty when none).

end Memcp.Json;
