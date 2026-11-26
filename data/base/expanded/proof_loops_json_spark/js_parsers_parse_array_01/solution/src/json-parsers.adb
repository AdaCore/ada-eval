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

pragma Warnings (Off, "literal ""Duplicate_Key"" is not referenced");
pragma Warnings (Off, "literal ""Expected_Name_Separator"" is not referenced");
pragma Warnings (Off, "literal ""Expected_String_Key"" is not referenced");
with JSON.Streams;
with JSON.Tokenizers;

package body JSON.Parsers
  with SPARK_Mode => On
is

   package Tokenizers is new JSON.Tokenizers (Types);

   use type Tokenizers.Token_Kind;
   use type Tokenizers.Token_Status;
   use type Types.Value_Kind;

   type Parse_Status is
     (Ok,
      Tokenizer_Failed,
      Unexpected_Token,
      Expected_Value_Separator,
      Expected_Name_Separator,
      Expected_String_Key,
      Duplicate_Key,
      Depth_Exceeded,
      Too_Many_Elements);

   --  Parse_Value, Parse_Array, and Parse_Object report failures via
   --  Status instead of raising an exception, so that each level of the
   --  recursion can release the partial tree it owns; procedure Parse
   --  raises Parse_Error at the top

   procedure Parse_Value
     (Stream        : in out Streams.Stream;
      Current       : Tokenizers.Token;
      Depth         : Positive;
      Maximum_Depth : Positive;
      Result        : out Types.JSON_Value_Access;
      Status        : out Parse_Status;
      Detail        : out Tokenizers.Token_Status)
   with
     Pre                =>
       Depth <= Maximum_Depth
       and then
         (if Current.Kind = Tokenizers.String_Token
          then
            Streams.Valid_Span
              (Stream, Current.String_Offset, Current.String_Length)),
     Post               =>
       Streams.Length (Stream) = Streams.Length (Stream)'Old
       and then Streams.Position (Stream) >= Streams.Position (Stream)'Old
       and then
         (if Streams.Has_Buffered_Character (Stream)
            and then Streams.Position (Stream) = Streams.Position (Stream)'Old
          then Streams.Has_Buffered_Character (Stream)'Old)
       and then
         (if Status = Ok
          then Result /= null and then Types.Is_Standalone (Result)
          else Result = null),
     Always_Terminates,
     Subprogram_Variant =>
       (Decreases => Maximum_Depth - Depth, Decreases => 0);

   procedure Parse_Array
     (Stream        : in out Streams.Stream;
      Depth         : Positive;
      Maximum_Depth : Positive;
      Result        : out Types.JSON_Value_Access;
      Status        : out Parse_Status;
      Detail        : out Tokenizers.Token_Status)
   with
     Pre                => Depth <= Maximum_Depth,
     Post               =>
       Streams.Length (Stream) = Streams.Length (Stream)'Old
       and then Streams.Position (Stream) >= Streams.Position (Stream)'Old
       and then
         (if Streams.Has_Buffered_Character (Stream)
            and then Streams.Position (Stream) = Streams.Position (Stream)'Old
          then Streams.Has_Buffered_Character (Stream)'Old)
       and then
         (if Status = Ok
          then Result /= null and then Types.Is_Standalone (Result)
          else Result = null),
     Always_Terminates,
     Subprogram_Variant =>
       (Decreases => Maximum_Depth - Depth, Decreases => 1);

   procedure Parse_Object
     (Stream        : in out Streams.Stream;
      Depth         : Positive;
      Maximum_Depth : Positive;
      Result        : out Types.JSON_Value_Access;
      Status        : out Parse_Status;
      Detail        : out Tokenizers.Token_Status)
   with
     Pre                => Depth <= Maximum_Depth,
     Post               =>
       Streams.Length (Stream) = Streams.Length (Stream)'Old
       and then Streams.Position (Stream) >= Streams.Position (Stream)'Old
       and then
         (if Streams.Has_Buffered_Character (Stream)
            and then Streams.Position (Stream) = Streams.Position (Stream)'Old
          then Streams.Has_Buffered_Character (Stream)'Old)
       and then
         (if Status = Ok
          then Result /= null and then Types.Is_Standalone (Result)
          else Result = null),
     Import,
     Global             => null,
     Always_Terminates,
     Subprogram_Variant =>
       (Decreases => Maximum_Depth - Depth, Decreases => 1);

   procedure Parse_Value
     (Stream        : in out Streams.Stream;
      Current       : Tokenizers.Token;
      Depth         : Positive;
      Maximum_Depth : Positive;
      Result        : out Types.JSON_Value_Access;
      Status        : out Parse_Status;
      Detail        : out Tokenizers.Token_Status) is
   begin
      Result := null;
      Status := Ok;
      Detail := Tokenizers.Success;

      case Current.Kind is
         when Tokenizers.Begin_Array_Token  =>
            if Depth >= Maximum_Depth then
               Status := Depth_Exceeded;
            else
               Parse_Array
                 (Stream, Depth + 1, Maximum_Depth, Result, Status, Detail);
            end if;

         when Tokenizers.Begin_Object_Token =>
            if Depth >= Maximum_Depth then
               Status := Depth_Exceeded;
            else
               Parse_Object
                 (Stream, Depth + 1, Maximum_Depth, Result, Status, Detail);
            end if;

         when Tokenizers.String_Token       =>
            Result :=
              Types.Create_String
                (Streams.Get_String
                   (Stream, Current.String_Offset, Current.String_Length));

         when Tokenizers.Integer_Token      =>
            Result := Types.Create_Integer (Current.Integer_Value);

         when Tokenizers.Float_Token        =>
            Result := Types.Create_Float (Current.Float_Value);

         when Tokenizers.Boolean_Token      =>
            Result := Types.Create_Boolean (Current.Boolean_Value);

         when Tokenizers.Null_Token         =>
            Result := Types.Create_Null;

         when others                        =>
            Status := Unexpected_Token;
      end case;
   end Parse_Value;

   procedure Parse_Array
     (Stream        : in out Streams.Stream;
      Depth         : Positive;
      Maximum_Depth : Positive;
      Result        : out Types.JSON_Value_Access;
      Status        : out Parse_Status;
      Detail        : out Tokenizers.Token_Status)
   is separate;

end JSON.Parsers;
