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

separate (JSON.Tokenizers)
procedure Scan_Number_Exponent_Digits
  (Stream     : in out Streams.Stream;
   Char_Count : in out Natural;
   Int_Acc    : Base_Integer;
   F_Acc      : Base_Float;
   Exp_Value  : in out Natural;
   Status     : out Token_Status)
is
   C   : Character;
   EOF : Boolean;

   function To_Digit (Value : Character) return Base_Integer
   is (Character'Pos (Value) - Character'Pos ('0'))
   with Pre => Value in '0' .. '9';
begin
   Status := Success;

   --  Accept a sequence of digits
   loop
      pragma Loop_Invariant (not Streams.Has_Buffered_Character (Stream));
      pragma
        Loop_Invariant
          (Streams.Length (Stream) = Streams.Length (Stream)'Loop_Entry);
      pragma
        Loop_Invariant
          (Int_Acc <= 0 and F_Acc in 0.0 .. Mantissa_Limit * 10.0 + 9.0);
      pragma
        Loop_Invariant (Char_Count <= Types.Maximum_String_Length_Numbers);
      pragma Loop_Invariant (Exp_Value <= Exp_Limit);
      pragma
        Loop_Invariant
          (Streams.Position (Stream)
           >= Streams.Position (Stream)'Loop_Entry);
      pragma Loop_Variant (Increases => Streams.Position (Stream));

      Streams.Read_Character (Stream, C, EOF);
      exit when EOF or else C not in '0' .. '9';

      if Char_Count = Types.Maximum_String_Length_Numbers then
         Status := Error_Number_Too_Long;
         return;
      end if;
      Char_Count := Char_Count + 1;

      if Exp_Value <= (Exp_Limit - 9) / 10 then
         Exp_Value := Exp_Value * 10 + Natural (To_Digit (C));
      else
         Exp_Value := Exp_Limit;
      end if;
   end loop;
end Scan_Number_Exponent_Digits;
