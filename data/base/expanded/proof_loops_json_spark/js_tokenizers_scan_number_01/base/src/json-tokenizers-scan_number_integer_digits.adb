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
procedure Scan_Number_Integer_Digits
  (Stream       : in out Streams.Stream;
   First        : Character;
   Char_Count   : out Natural;
   Int_Acc      : out Base_Integer;
   Int_Overflow : out Boolean;
   F_Acc        : out Base_Float;
   F_Overflow   : out Boolean;
   Status       : out Token_Status)
is
   C   : Character;
   EOF : Boolean;

   Is_Negative : constant Boolean := First = '-';

   Int_Digits  : Natural := 0;
   First_Digit : Character := '0';

   function To_Digit (Value : Character) return Base_Integer
   is (Character'Pos (Value) - Character'Pos ('0'))
   with Pre => Value in '0' .. '9';

   procedure Add_Digit (Digit : Base_Integer)
   with
     Pre  =>
       Digit in 0 .. 9
       and then Int_Acc <= 0
       and then F_Acc in 0.0 .. Mantissa_Limit * 10.0 + 9.0,
     Always_Terminates,
     Post =>
       Int_Acc <= 0 and then F_Acc in 0.0 .. Mantissa_Limit * 10.0 + 9.0;

   procedure Add_Digit (Digit : Base_Integer) is
   begin
      if not Int_Overflow then
         if Int_Acc < (Base_Integer'First + Digit) / 10 then
            Int_Overflow := True;
         else
            Int_Acc := Int_Acc * 10 - Digit;
         end if;
      end if;

      if not F_Overflow then
         if F_Acc > Mantissa_Limit then
            F_Overflow := True;
         else
            F_Acc := F_Acc * 10.0 + Base_Float (Digit);
         end if;
      end if;
   end Add_Digit;
begin
   Status := Success;

   Int_Acc := 0;     --  Accumulated as a non-positive value
   Int_Overflow := False;

   F_Acc := 0.0;     --  Accumulated absolute mantissa
   F_Overflow := False;

   Char_Count := 1;

   if First = '+' then
      Status := Error_Plus_Sign;
      return;
   end if;

   if not Is_Negative then
      First_Digit := First;
      Int_Digits := 1;
      Add_Digit (To_Digit (First));
   end if;

   --  Accept a sequence of digits for the integer part
   loop
      pragma Loop_Variant (Increases => Streams.Position (Stream));

      Streams.Read_Character (Stream, C, EOF);
      exit when EOF or else C not in '0' .. '9';

      if Char_Count = Types.Maximum_String_Length_Numbers then
         Status := Error_Number_Too_Long;
         return;
      end if;
      Char_Count := Char_Count + 1;

      if Int_Digits = 0 then
         First_Digit := C;
      end if;
      Int_Digits := Int_Digits + 1;
      Add_Digit (To_Digit (C));
   end loop;

   --  Test for a missing digit after a '-' sign and for leading zeroes
   if Int_Digits = 0 then
      Status := Error_Minus_Digit;
      return;
   elsif First_Digit = '0' and then Int_Digits >= 2 then
      Status := Error_Leading_Zeroes;
      return;
   end if;
end Scan_Number_Integer_Digits;
