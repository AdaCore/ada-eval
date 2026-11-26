--  SPDX-License-Identifier: Apache-2.0
--
--  Copyright (c) 2016 onox <denkpadje@gmail.com>
--
--  Licensed under the Apache License, Version 2.0 (the "License");
--  you may not use this file except in compliance with the License.
--  You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
--  Unless required by applicable law or agreed to in writing, software
--  distributed under the License is distributed on an "AS IS" BASIS,
--  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
--  See the License for the specific language governing permissions and
--  limitations under the License.

--  Extracted from json-spark, json/src/json-types.ads.
--
--  Trimmed to the operations Parse_Object and Parse_Value use.  Kind, Length,
--  Is_Standalone and the six Create_* functions are expression functions
--  upstream and are copied whole (their bodies are their meaning); Prepend,
--  Reverse_Elements and Free have real bodies and are imported as specs, their
--  contracts unchanged apart from the Global and Depends that upstream let
--  GNATprove derive from those bodies.  The JSON_Value record keeps all nine
--  components.

pragma Warnings (Off, "use clause for type ""Text_Access"" defined at * has*");
with JSON.Streams;

generic
   type Integer_Type is range <>;
   type Float_Type is digits <>;
package JSON.Types with SPARK_Mode => On is
   --  JSON values and the operations to inspect and build them.
   --  @formal Integer_Type The integer type used for JSON integer values
   --  @formal Float_Type The floating-point type used for JSON float
   --    values

   pragma Preelaborate;

   type Value_Kind is
     (Array_Kind,
      Object_Kind,
      String_Kind,
      Integer_Kind,
      Float_Kind,
      Boolean_Kind,
      Null_Kind);
   --  The kind of data held by a JSON value
   --  @enum Array_Kind A JSON array
   --  @enum Object_Kind A JSON object
   --  @enum String_Kind A JSON string
   --  @enum Integer_Kind A JSON number with no fractional part or exponent
   --  @enum Float_Kind A JSON number with a fractional part or exponent
   --  @enum Boolean_Kind A JSON boolean (true or false)
   --  @enum Null_Kind The JSON null value

   subtype Composite_Kind is Value_Kind range Array_Kind .. Object_Kind;
   --  The kinds that hold elements or members: arrays and objects

   type JSON_Value is limited private;
   --  A node of a JSON document. Values are always handled through
   --  access values: owning JSON_Value_Access objects returned by the
   --  constructors below and by Parsers.Parse, and observers
   --  (access constant JSON_Value) returned by Get, First, and Next.

   type JSON_Value_Access is access JSON_Value;
   --  Owning access to a JSON value. The value owns its elements or
   --  members, so calling Free on the root of a document releases the
   --  whole document.

   ----------------------------------------------------------------------------
   --                                Observers
   --                                --
   ----------------------------------------------------------------------------

   function Kind
     (Object : not null access constant JSON_Value) return Value_Kind;
   --  Return the kind of data held by the JSON value
   --  @param Object The JSON value to inspect
   --  @return The kind of the JSON value

   function Length
     (Object : not null access constant JSON_Value) return Natural
   with Pre => Kind (Object) in Composite_Kind;
   --  Return the number of elements or members of a JSON array or object
   --  @param Object The JSON value, which must be an array or object
   --  @return The number of elements or members

   function Contains
     (Object : not null access constant JSON_Value; Key : String)
      return Boolean
   with Pre => Kind (Object) = Object_Kind, Import, Global => null;
   --  Return True if the JSON object contains a member for the given key
   --
   --  This function has a time complexity of O(n).
   --  @param Object The JSON value, which must be of Object_Kind
   --  @param Key The member key to look for, in JSON escaped form
   --  @return True if the object has a member with the given key

   ----------------------------------------------------------------------------
   --                              Constructors
   --                              --
   ----------------------------------------------------------------------------

   --  The objects returned by these functions own heap memory; either
   --  transfer ownership with Append, Insert, Prepend, or Prepend_Member,
   --  or call Free

   function Is_Standalone
     (Object : not null access constant JSON_Value) return Boolean;
   --  Return True if the value is not an element or member of an array
   --  or object. Values returned by the constructors below are
   --  standalone; the building procedures require and consume
   --  standalone values.
   --  @param Object The JSON value to query
   --  @return True if the value is not part of any array or object

   function Create_String (Value : String) return JSON_Value_Access
   with
     Post =>
       Create_String'Result /= null
       and then Kind (Create_String'Result) = String_Kind
       and then Is_Standalone (Create_String'Result);
   --  Create a JSON string value from a string in JSON escaped form
   --  (the form in which strings appear in JSON text, without the
   --  surrounding '"' characters)
   --  @param Value The string, in JSON escaped form
   --  @return A new standalone JSON string value owned by the caller

   function Create_Integer (Value : Integer_Type) return JSON_Value_Access
   with
     Post =>
       Create_Integer'Result /= null
       and then Kind (Create_Integer'Result) = Integer_Kind
       and then Is_Standalone (Create_Integer'Result);
   --  Create a JSON integer value
   --  @param Value The integer value
   --  @return A new standalone JSON integer value owned by the caller

   function Create_Float (Value : Float_Type) return JSON_Value_Access
   with
     Post =>
       Create_Float'Result /= null
       and then Kind (Create_Float'Result) = Float_Kind
       and then Is_Standalone (Create_Float'Result);
   --  Create a JSON float value
   --  @param Value The floating-point value
   --  @return A new standalone JSON float value owned by the caller

   function Create_Boolean (Value : Boolean) return JSON_Value_Access
   with
     Post =>
       Create_Boolean'Result /= null
       and then Kind (Create_Boolean'Result) = Boolean_Kind
       and then Is_Standalone (Create_Boolean'Result);
   --  Create a JSON boolean value
   --  @param Value The boolean value
   --  @return A new standalone JSON boolean value owned by the caller

   function Create_Null return JSON_Value_Access
   with
     Post =>
       Create_Null'Result /= null
       and then Kind (Create_Null'Result) = Null_Kind
       and then Is_Standalone (Create_Null'Result);
   --  Create a JSON null value
   --  @return A new standalone JSON null value owned by the caller

   function Create_Object return JSON_Value_Access
   with
     Post =>
       Create_Object'Result /= null
       and then Kind (Create_Object'Result) = Object_Kind
       and then Length (Create_Object'Result) = 0
       and then Is_Standalone (Create_Object'Result);
   --  Create an empty JSON object; fill it with Insert or Prepend_Member
   --  @return A new standalone, empty JSON object owned by the caller

   ----------------------------------------------------------------------------
   --                                Building
   --                                --
   ----------------------------------------------------------------------------

   procedure Prepend_Member
     (Object : not null access JSON_Value;
      Key    : String;
      Value  : in out JSON_Value_Access)
   with
     Pre     =>
       Kind (Object) = Object_Kind
       and then Value /= null
       and then Is_Standalone (Value)
       and then Length (Object) < Natural'Last,
     Import,
     Global  => null,
     Depends => (Object => (Object, Key, Value), Value => null),
     Always_Terminates,
     Post    =>
       Value = null
       and then Kind (Object) = Object_Kind
       and then Length (Object) = Length (Object)'Old + 1
       and then Is_Standalone (Object) = Is_Standalone (Object)'Old;
   --  Like Insert, but adds the member to the front in O(1) time
   --  @param Object The JSON object to prepend the member to
   --  @param Key The member key, in JSON escaped form
   --  @param Value The standalone value to add; set to null on return as
   --    ownership is transferred to Object

   procedure Reverse_Elements (Object : not null access JSON_Value)
   with
     Pre    => Kind (Object) in Composite_Kind,
     Import,
     Global => null,
     Always_Terminates,
     Post   =>
       Kind (Object) = Kind (Object)'Old
       and then Length (Object) = Length (Object)'Old
       and then Is_Standalone (Object) = Is_Standalone (Object)'Old;
   --  Reverse the order of the elements or members of a JSON array or
   --  object. Building a container with Prepend or Prepend_Member and
   --  then calling Reverse_Elements once is equivalent to building it
   --  with Append or Insert, in O(n) total time.
   --  @param Object The JSON array or object whose elements are reversed

   procedure Free (Object : in out JSON_Value_Access)
   with
     Import,
     Global  => null,
     Always_Terminates,
     Post    => Object = null,
     Depends => (Object => null, null => Object);
   --  Release the value and everything it owns
   --  @param Object The value to release; set to null on return (a null
   --    value is accepted and is a no-op)

private

   use type Streams.Text_Access;

   type JSON_Value is limited record
      Kind          : Value_Kind := Null_Kind;
      Next          : JSON_Value_Access := null;
      Key           : Streams.Text_Access := null;
      Str           : Streams.Text_Access := null;
      Boolean_Value : Boolean := False;
      Integer_Value : Integer_Type := Integer_Type'First;
      Float_Value   : Float_Type := Float_Type'First;
      First_Child   : JSON_Value_Access := null;
      Length        : Natural := 0;
   end record;
   --  Full view of JSON_Value: a discriminant-free record holding the payload
   --  of every kind, plus the two links that make a composite value's
   --  elements a singly-linked list it owns -- First_Child down into the list
   --  and Next along it. Only the components Kind selects are meaningful; the
   --  rest keep their defaults.
   --
   --  A single record rather than a variant is what lets a node be built
   --  before its kind is known and lets Reverse_Elements relink a list in
   --  place, and it keeps the ownership structure GNATprove tracks to the two
   --  access components.
   --
   --  Str is never null when Kind = String_Kind for values built with
   --  the constructors; the observers nevertheless treat a null Str as
   --  an empty string, because a type predicate stating the property
   --  cannot be re-established by GNATprove when a container is
   --  reconstructed at the end of a borrow.
   --  @field Kind Which of the payload components below is meaningful
   --  @field Next The next sibling in the list of elements or members of the
   --    composite value that owns this node, or null at the end of the list
   --  @field Key The key of this node when it is a member of a JSON object,
   --    in escaped form; null for array elements and for standalone values
   --  @field Str The text of a String_Kind value, in escaped form
   --  @field Boolean_Value The value of a Boolean_Kind value
   --  @field Integer_Value The value of an Integer_Kind value
   --  @field Float_Value The value of a Float_Kind value
   --  @field First_Child The first element or member owned by an
   --    Array_Kind or Object_Kind value, or null when it is empty
   --  @field Length The number of elements or members of an Array_Kind or
   --    Object_Kind value

end JSON.Types;
