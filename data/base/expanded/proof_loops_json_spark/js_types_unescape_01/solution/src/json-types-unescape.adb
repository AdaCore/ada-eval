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

separate (JSON.Types)
function Unescape (Text : String) return String is
   use Ada.Characters.Latin_1;

   Result : String (1 .. Text'Length) := (others => ' ');
   Last   : Natural := 0;

   Skip : Natural := 0;
   --  Number of characters of Text already consumed as part of a
   --  multi-character escape sequence and still to be stepped over

   function To_Hex_Digit (Value : Character) return Natural
   is (case Value is
         when '0' .. '9' => Character'Pos (Value) - Character'Pos ('0'),
         when 'a' .. 'f' =>
           Character'Pos (Value) - Character'Pos ('a') + 10,
         when 'A' .. 'F' =>
           Character'Pos (Value) - Character'Pos ('A') + 10,
         when others     => 0)
   with Post => To_Hex_Digit'Result <= 15;

   function Is_Hex_Quad_At (Index : Positive) return Boolean
   is (Index in Text'Range
       and then Text'Last - Index >= 3
       and then
         (for all J in Index .. Index + 3 =>
            Text (J) in '0' .. '9' | 'a' .. 'f' | 'A' .. 'F'));
   --  True if Text has 4 hexadecimal digits starting at Index

   function Hex_Quad_At (Index : Positive) return Natural
   is (((To_Hex_Digit (Text (Index)) * 16 + To_Hex_Digit (Text (Index + 1)))
        * 16
        + To_Hex_Digit (Text (Index + 2)))
       * 16
       + To_Hex_Digit (Text (Index + 3)))
   with
     Pre  => Is_Hex_Quad_At (Index),
     Post => Hex_Quad_At'Result <= 16#FFFF#;
   --  Value of the 4 hexadecimal digits starting at Index, as a
   --  UTF-16 code unit

   procedure Append_Code_Point (Code : Natural)
   with
     Pre  => Code <= 16#10FFFF# and then Last <= Result'Length - 4,
     Post =>
       Last in Last'Old + 1 .. Last'Old + 4
       and then (if Code <= 16#FFFF# then Last <= Last'Old + 3);
   --  Append the UTF-8 encoding (1 to 4 bytes) of a code point

   procedure Append_Code_Point (Code : Natural) is
   begin
      if Code <= 16#7F# then
         Result (Last + 1) := Character'Val (Code);
         Last := Last + 1;
      elsif Code <= 16#7FF# then
         Result (Last + 1) := Character'Val (16#C0# + Code / 64);
         Result (Last + 2) := Character'Val (16#80# + Code mod 64);
         Last := Last + 2;
      elsif Code <= 16#FFFF# then
         Result (Last + 1) := Character'Val (16#E0# + Code / 4096);
         Result (Last + 2) := Character'Val (16#80# + (Code / 64) mod 64);
         Result (Last + 3) := Character'Val (16#80# + Code mod 64);
         Last := Last + 3;
      else
         Result (Last + 1) := Character'Val (16#F0# + Code / 262144);
         Result (Last + 2) := Character'Val (16#80# + (Code / 4096) mod 64);
         Result (Last + 3) := Character'Val (16#80# + (Code / 64) mod 64);
         Result (Last + 4) := Character'Val (16#80# + Code mod 64);
         Last := Last + 4;
      end if;
   end Append_Code_Point;
begin
   for J in Text'Range loop
      pragma Loop_Invariant (Last <= J - Text'First + Skip);
      pragma Loop_Invariant (Skip <= Text'Last - J + 1);

      if Skip > 0 then
         Skip := Skip - 1;
      else
         declare
            C : constant Character := Text (J);
         begin
            if C = '\' and then J < Text'Last then
               declare
                  E : constant Character := Text (J + 1);
               begin
                  if E = 'u'
                    and then Text'Last - J >= 5
                    and then Is_Hex_Quad_At (J + 2)
                  then
                     declare
                        Unit : constant Natural := Hex_Quad_At (J + 2);
                     begin
                        if Unit in 16#D800# .. 16#DBFF#
                          and then Text'Last - J >= 11
                          and then Text (J + 6) = '\'
                          and then Text (J + 7) = 'u'
                          and then Is_Hex_Quad_At (J + 8)
                          and then
                            Hex_Quad_At (J + 8) in 16#DC00# .. 16#DFFF#
                        then
                           --  A UTF-16 surrogate pair encoding a
                           --  code point beyond the Basic
                           --  Multilingual Plane
                           Append_Code_Point
                             (16#10000#
                              + (Unit - 16#D800#) * 16#400#
                              + (Hex_Quad_At (J + 8) - 16#DC00#));
                           Skip := 11;
                        elsif Unit in 16#D800# .. 16#DFFF# then
                           --  A lone surrogate is not a valid code
                           --  point; the tokenizer rejects it, but
                           --  emit U+FFFD (replacement character)
                           --  to keep this function total
                           Append_Code_Point (16#FFFD#);
                           Skip := 5;
                        else
                           Append_Code_Point (Unit);
                           Skip := 5;
                        end if;
                     end;
                  else
                     --  A single-character escape sequence (or a
                     --  malformed \u escape, which the tokenizer
                     --  rejects; the character is copied unchanged
                     --  to keep this function total)
                     Last := Last + 1;
                     Result (Last) :=
                       (case E is
                          when 'b'    => BS,
                          when 'f'    => FF,
                          when 'n'    => LF,
                          when 'r'    => CR,
                          when 't'    => HT,
                          when others => E);
                     Skip := 1;
                  end if;
               end;
            elsif C = '\' then
               --  A trailing \ at the end of the text, which the
               --  tokenizer rejects; skip it to keep this function
               --  total
               null;
            else
               Last := Last + 1;
               Result (Last) := C;
            end if;
         end;
      end if;
   end loop;
   return Result (1 .. Last);
end Unescape;
