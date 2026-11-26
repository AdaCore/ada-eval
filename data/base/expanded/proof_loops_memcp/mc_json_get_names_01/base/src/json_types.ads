--  An opaque stand-in for the json-spark value model that Memcp.Json's body
--  instantiates as `Types`.  json-spark is not part of this project,
--  so the entities the loop names are imported with no bodies: the kind
--  enumeration and Composite_Kind verbatim from JSON.Types, JSON_Value as a
--  private type with nothing observable in it, and Kind / Value / Length / Get
--  with the preconditions JSON.Types states for them.  Nothing here is
--  stronger than upstream: the postconditions upstream does not write are not
--  written here either.

package Json_Types
  with SPARK_Mode => On
is

   type Value_Kind is
     (Array_Kind,
      Object_Kind,
      String_Kind,
      Integer_Kind,
      Float_Kind,
      Boolean_Kind,
      Null_Kind);
   --  The kind of data held by a JSON value

   subtype Composite_Kind is Value_Kind range Array_Kind .. Object_Kind;
   --  The kinds that hold elements or members: arrays and objects

   type JSON_Value is limited private;
   --  A node of a JSON document, handled only through observers.

   function Kind
     (Object : not null access constant JSON_Value) return Value_Kind
   with Import, Global => null;
   --  Return the kind of data held by the JSON value

   function Value (Object : not null access constant JSON_Value) return String
   with Import, Global => null, Pre => Kind (Object) = String_Kind;
   --  Return the string held by a JSON string value

   function Length
     (Object : not null access constant JSON_Value) return Natural
   with Import, Global => null, Pre => Kind (Object) in Composite_Kind;
   --  Return the number of elements or members of a JSON array or object

   function Get
     (Object : not null access constant JSON_Value; Index : Positive)
      return access constant JSON_Value
   with Import, Global => null, Pre => Kind (Object) = Array_Kind;
   --  Return the JSON value at the given index in the JSON array, or null if
   --  the array has no such index

private

   type JSON_Value is limited record
      Opaque : Boolean;
      --  A placeholder: the real representation is a discriminated node that
      --  this loop never inspects.
   end record;

end Json_Types;
