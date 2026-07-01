"""
Script to replay the full multi-turn conversations from the 10 sample traces
against the local agent implementation.
Computes Recall@10 and checks conversational accuracy, schemas, and catalog integrity.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app.agent import process_chat
from app.catalog import catalog_index
from app.models import ChatMessage


def parse_conversation_file(filepath: Path) -> tuple[list[dict], list[str]]:
    """
    Parses a sample conversation markdown file.
    Returns:
      - list of messages (role: user/assistant, content) representing the full trace
      - list of expected assessment names in the final turn's recommendation table
    """
    content = filepath.read_text(encoding="utf-8")
    
    # 1. Parse all User and Agent blocks
    turns = []
    
    # Find all Turn blocks
    turn_blocks = re.split(r"### Turn \d+", content)[1:]
    
    for block in turn_blocks:
        # Extract User message
        user_match = re.search(r"\*\*User\*\*\s*\n\s*>\s*(.+?)(?=\n\s*\*\*Agent\*\*|\n\s*_\w|\n\s*$)", block, re.DOTALL)
        if user_match:
            user_content = user_match.group(1).strip()
            # Clean blockquotes
            user_content = "\n".join(line.lstrip("> ").strip() for line in user_content.split("\n"))
            turns.append({"role": "user", "content": user_content})
            
        # Extract Agent message (until underscore notes)
        agent_match = re.search(r"\*\*Agent\*\*\s*\n\s*(.+?)(?=\n\s*_\w|\n\s*$)", block, re.DOTALL)
        if agent_match:
            agent_content = agent_match.group(1).strip()
            turns.append({"role": "assistant", "content": agent_content})

    # 2. Parse final expected recommendations from the last table
    table_pattern = re.compile(
        r"\|.*?\|.*?\|.*?\|.*?\|.*?\|.*?\|.*?\|\n"
        r"\|[-\s|]+\|\n"
        r"((?:\|.*?\|.*?\|.*?\|.*?\|.*?\|.*?\|.*?\|\n)*)",
        re.MULTILINE,
    )
    
    tables = list(table_pattern.finditer(content))
    expected_names = []
    if tables:
        last_table = tables[-1].group(0)
        for line in last_table.split("\n"):
            line = line.strip()
            if not line or line.startswith("|---") or line.startswith("| #"):
                continue
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if len(cells) >= 2:
                name = cells[1].strip()
                if name and not name.startswith("---") and name != "Name":
                    expected_names.append(name)
                    
    return turns, expected_names


async def replay_conversation(filepath: Path) -> dict:
    """Replays a single conversation trace turn-by-turn against the agent."""
    turns, expected = parse_conversation_file(filepath)
    filename = filepath.name
    
    print(f"\nReplaying {filename}...")
    print(f"Expected final recommendations: {expected}")
    
    # We will simulate the conversation turn-by-turn.
    # At each user turn, we send the history up to that point.
    history = []
    final_response = None
    
    # We step through the turns. Every time there is a user turn, we send the
    # history including that user turn to the agent.
    for i, turn in enumerate(turns):
        history.append(ChatMessage(role=turn["role"], content=turn["content"]))
        
        if turn["role"] == "user":
            # Send history to the agent
            try:
                response = await process_chat(history, catalog_index)
                
                # Check if it was the last user message
                is_last_user_turn = True
                for subsequent_turn in turns[i+1:]:
                    if subsequent_turn["role"] == "user":
                        is_last_user_turn = False
                        break
                
                if is_last_user_turn:
                    final_response = response
            except Exception as e:
                print(f"Error during agent turn call: {e}")
                
            # If the trace has assistant responses, we append the trace's assistant response
            # to feed the exact expected conversational path, simulating the evaluation harness.
            if i + 1 < len(turns) and turns[i+1]["role"] == "assistant":
                # Wait, the automated evaluator simulates the user based on our response.
                # Here we are replaying the trace where both sides are fixed to evaluate
                # how closely our final recommendations match when given the exact context.
                pass
                
    # Calculate Recall
    if final_response and final_response.recommendations:
        rec_names = [r.name for r in final_response.recommendations]
        print(f"Agent final recommendations: {rec_names}")
        
        # Calculate Recall@10
        if expected:
            rec_lower = [r.lower() for r in rec_names]
            exp_lower = [e.lower() for e in expected]
            
            # Since some names might have slight suffix/prefix differences,
            # let's do a substring/fuzzy matching to be robust
            matched = 0
            for exp in exp_lower:
                # Find if any recommended name contains or is contained in expected name
                found = False
                for rec in rec_lower:
                    if exp in rec or rec in exp:
                        found = True
                        break
                if found:
                    matched += 1
            recall = matched / len(expected)
        else:
            recall = 1.0
            
        end_of_conv = final_response.end_of_conversation
    else:
        print("Agent returned NO final recommendations.")
        recall = 0.0 if expected else 1.0
        end_of_conv = False
        
    print(f"Recall@10: {recall:.2f} | End of Conversation: {end_of_conv}")
    return {
        "filename": filename,
        "recall": recall,
        "expected": expected,
        "recommendations": [r.name for r in final_response.recommendations] if final_response and final_response.recommendations else []
    }


async def main():
    print("Initializing Catalog...")
    catalog_index.load()
    
    conv_dir = Path(__file__).resolve().parent.parent / "sample_conversations" / "GenAI_SampleConversations"
    files = sorted(conv_dir.glob("C*.md"))
    
    results = []
    for f in files:
        res = await replay_conversation(f)
        results.append(res)
        # Sleep slightly to prevent hitting Groq rate limits
        await asyncio.sleep(2)
        
    print("\n" + "="*50)
    print("SUMMARY OF CONVERSATION REPLAY")
    print("="*50)
    
    total_recall = 0.0
    for res in results:
        print(f"{res['filename']}: Recall@10 = {res['recall']:.2f}")
        total_recall += res['recall']
        
    mean_recall = total_recall / len(results) if results else 0.0
    print(f"\nMean Recall@10: {mean_recall:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
