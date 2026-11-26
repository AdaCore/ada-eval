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

--  Extracted from json-spark, json/src/json-streams.ads.
--
--  Get_String is imported as a spec with its contract unchanged; Parse_Value
--  calls it to copy a string token's span out of the text.
--
--  What the isolation dropped: the String_Buffer half of the package, which no
--  parser loop touches, and Has_Storage / From_Text / From_File / Destroy /
--  Read_Character / Write_Character, none of which anything kept here calls --
--  the stream is only ever advanced through Tokenizers.Scan_Token, which is
--  imported.  Text_Access stays because the JSON_Value record has two
--  components of that type.

package JSON.Streams
  with SPARK_Mode => On
is
   pragma Preelaborate;

   type Text_Access is access String;
   --  Access type designating a heap-allocated String, used to hold the
   --  text owned by a String_Buffer or a Stream

   type Stream is limited private;
   --  A character stream over a text held in heap memory. An object of
   --  this type owns the text; call Destroy to release it. A stream
   --  without text behaves as a stream over an empty text.

   function Length (Object : Stream) return Natural
   with Post => Length'Result < Positive'Last;
   --  Length of the text of the stream
   --
   --  @param Object The stream to query
   --  @return The number of characters in the stream's text

   function Position (Object : Stream) return Positive
   with Post => Position'Result <= Length (Object) + 1;
   --  Index of the next character to be read from the text
   --
   --  @param Object The stream to query
   --  @return The one-based index of the next character to be read

   function Has_Buffered_Character (Object : Stream) return Boolean;
   --  Return whether a character has been pushed back into the stream by
   --  Write_Character and not yet consumed
   --
   --  @param Object The stream to query
   --  @return True if a buffered character is pending

   function Valid_Span
     (Object : Stream; Offset : Positive; Count : Natural) return Boolean
   is (Count = 0
       or else
         (Offset <= Length (Object)
          and then Count <= Length (Object) - Offset + 1));
   --  True if the given span designates characters of the text
   --
   --  @param Object The stream whose text the span refers to
   --  @param Offset The one-based index of the first character of the span
   --  @param Count The number of characters in the span
   --  @return True if the span lies within the stream's text
   --    (an empty span is always valid)

   function Get_String
     (Object : Stream; Offset : Positive; Count : Natural) return String
   with
     Pre    => Valid_Span (Object, Offset, Count),
     Import,
     Global => null,
     Post   =>
       Get_String'Result'Length = Count and Get_String'Result'First = 1;
   --  Return a copy of the designated characters of the text
   --
   --  @param Object The stream whose text is copied
   --  @param Offset The one-based index of the first character to copy
   --  @param Count The number of characters to copy
   --  @return A String of length Count, indexed from 1, holding the copy

private

   type Stream is limited record
      Text           : Text_Access := null;
      Index          : Positive := 1;
      Next_Character : Character := ' ';
      Has_Next       : Boolean := False;
   end record
   with
     Dynamic_Predicate =>
       (if Stream.Text = null
        then Stream.Index = 1
        else
          Stream.Text'First = 1
          and then Stream.Text'Last < Positive'Last
          and then Stream.Index <= Stream.Text'Last + 1);
   --  Full view of Stream: an optionally-owned text, a read cursor into it,
   --  and a one-character pushback slot.
   --
   --  Text is the ownership handle the SPARK proof tracks: null means the
   --  stream owns nothing, so Destroy is idempotent and a defaulted object
   --  needs no finalization. The predicate is what makes the cursor
   --  arithmetic in Next and Read_Character provably in range -- Index is
   --  allowed to reach Text'Last + 1 (one past the end, i.e. exhausted) and
   --  no further, and Text'Last is kept below Positive'Last so that
   --  Index + 1 cannot overflow.
   --  @field Text The heap-allocated JSON text, or null when the stream owns
   --    no memory
   --  @field Index One-based index of the next character to be read from
   --    Text; Text'Last + 1 once the text is exhausted
   --  @field Next_Character The character pushed back by Write_Character,
   --    meaningful only when Has_Next is True
   --  @field Has_Next True when Next_Character holds a pushed-back character
   --    that has not yet been consumed

   function Length (Object : Stream) return Natural
   is (if Object.Text = null then 0 else Object.Text'Last);
   --  Length of the text of the stream; zero when no text is held
   --
   --  @param Object The stream to query
   --  @return The number of characters in the stream's text

   function Position (Object : Stream) return Positive
   is (Object.Index);
   --  Index of the next character to be read from the text
   --
   --  @param Object The stream to query
   --  @return The one-based index of the next character to be read

   function Has_Buffered_Character (Object : Stream) return Boolean
   is (Object.Has_Next);
   --  Return whether a character has been pushed back and not yet consumed
   --
   --  @param Object The stream to query
   --  @return True if a buffered character is pending

end JSON.Streams;
