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

   ----------------------------------------------------------------------------
   --                              Constructors
   --                              --
   ----------------------------------------------------------------------------

   function Is_Standalone
     (Object : not null access constant JSON_Value) return Boolean;
   --  Return True if the value is not an element or member of an array
   --  or object. Values returned by the constructors below are
   --  standalone; the building procedures require and consume
   --  standalone values.
   --  @param Object The JSON value to query
   --  @return True if the value is not part of any array or object

   ----------------------------------------------------------------------------
   --                                Building
   --                                --
   ----------------------------------------------------------------------------

   procedure Append
     (Object : not null access JSON_Value; Value : in out JSON_Value_Access)
   with
     Pre  =>
       Kind (Object) = Array_Kind
       and then Value /= null
       and then Is_Standalone (Value)
       and then Length (Object) < Natural'Last,
     Always_Terminates,
     Post =>
       Value = null
       and then Kind (Object) = Array_Kind
       and then Length (Object) = Length (Object)'Old + 1
       and then Is_Standalone (Object) = Is_Standalone (Object)'Old;
   --  Add a value to the end of a JSON array, taking ownership of Value
   --
   --  This procedure has a time complexity of O(n).
   --  @param Object The JSON array to append to
   --  @param Value The standalone value to add; set to null on return as
   --    ownership is transferred to Object

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
