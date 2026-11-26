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
--  What the isolation dropped: the String_Buffer half of the package, which
--  the tokenizer never touches, and Get_String / From_Text / From_File /
--  Destroy / Has_Storage, which no tokenizer loop calls.  Read_Character (both
--  overloads) and Write_Character are copied as specs with `with Import`,
--  contracts unchanged; their bodies are never needed to prove a caller.

package JSON.Streams with SPARK_Mode => On is
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
     (Object : Stream;
      Offset : Positive;
      Count  : Natural) return Boolean
   is (Count = 0
         or else (Offset <= Length (Object)
                    and then Count <= Length (Object) - Offset + 1));
   --  True if the given span designates characters of the text
   --
   --  @param Object The stream whose text the span refers to
   --  @param Offset The one-based index of the first character of the span
   --  @param Count The number of characters in the span
   --  @return True if the span lies within the stream's text
   --    (an empty span is always valid)

   procedure Read_Character
     (Object : in out Stream;
      Result : out Character;
      EOF    : out Boolean)
   with Import,
        Global => null,
        Always_Terminates,
        Post =>
     Length (Object) = Length (Object)'Old
       and not Has_Buffered_Character (Object)
       and (if EOF then
              Position (Object) = Position (Object)'Old
                and Position (Object) = Length (Object) + 1
                and not Has_Buffered_Character (Object)'Old
            elsif Has_Buffered_Character (Object)'Old then
              Position (Object) = Position (Object)'Old
            else
              Position (Object)'Old <= Length (Object)
                and Position (Object) = Position (Object)'Old + 1);
   --  Return the buffered character if present, otherwise the character
   --  at the current position. EOF is True iff the text is exhausted.
   --
   --  @param Object The stream to read from
   --  @param Result The character that was read (meaningful only when EOF
   --    is False)
   --  @param EOF True iff the text is exhausted and no character was read

   procedure Read_Character
     (Object : in out Stream;
      Index  : out Positive;
      Result : out Character;
      EOF    : out Boolean)
   with Pre  => not Has_Buffered_Character (Object),
        Import,
        Global => null,
        Always_Terminates,
        Post =>
     Length (Object) = Length (Object)'Old
       and not Has_Buffered_Character (Object)
       and (if EOF then
              Position (Object) = Position (Object)'Old
                and Position (Object) = Length (Object) + 1
            else
              Index = Position (Object)'Old
                and Index <= Length (Object)
                and Position (Object) = Index + 1);
   --  Writes the index of the read character to Index. This is needed
   --  for string tokens.
   --
   --  @param Object The stream to read from
   --  @param Index The one-based index of the read character (meaningful
   --    only when EOF is False)
   --  @param Result The character that was read (meaningful only when EOF
   --    is False)
   --  @param EOF True iff the text is exhausted and no character was read

   procedure Write_Character (Object : in out Stream; Next : Character)
     with Pre  => not Has_Buffered_Character (Object),
          Import,
          Global => null,
          Always_Terminates,
          Post =>
     Has_Buffered_Character (Object)
       and Length (Object) = Length (Object)'Old
       and Position (Object) = Position (Object)'Old;
   --  Buffer one character; the next call to Read_Character returns it
   --
   --  @param Object The stream to push the character back into
   --  @param Next The character to buffer

private

   type Stream is limited record
      Text           : Text_Access := null;
      Index          : Positive    := 1;
      Next_Character : Character   := ' ';
      Has_Next       : Boolean     := False;
   end record
     with Dynamic_Predicate =>
       (if Stream.Text = null then
          Stream.Index = 1
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

   function Position (Object : Stream) return Positive is (Object.Index);
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
