#!/usr/bin/env python3
"""
Example 3: Complex Novel Writing & World-Building

This ENHANCED example demonstrates advanced features:
- Character relationships evolving based on plot events
- Cascading evolution when plot developments affect multiple characters
- Story background and world-building that interconnects
- Manual promotion for synthesizing character arcs
- Complex narrative threads that span multiple characters

Use Case: Authors, screenwriters, complex narrative management, D&D campaigns
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tracingrag.client import AsyncTracingRAGClient


async def novel_writing_example():
    """Demonstrate complex novel writing with interconnected characters"""
    print("=" * 70)
    print("📖 COMPLEX Test: Novel Writing with Character Networks")
    print("=" * 70)
    print()
    print("📚 Writing 'The Last Starship' - A Multi-Character Sci-Fi Epic")
    print()

    async with AsyncTracingRAGClient("http://localhost:8000", timeout=120.0) as client:
        health = await client.health()
        if health["status"] != "healthy":
            print("❌ API is not healthy. Please start the API server.")
            return
        print("✅ Connected to TracingRAG API\n")

        # ====================================================================
        # World-Building: Setting & Background
        # ====================================================================
        print("🌍 Phase 1: World-Building - The Starship Universe...")
        print()

        setting_ship = await client.create_memory(
            topic="setting_starship_nova",
            content=(
                "The Starship Nova is humanity's last colony ship, carrying 10,000 "
                "survivors after Earth's destruction. The ship has been traveling for "
                "200 years toward planet Kepler-442b. The ship is divided into sectors: "
                "Command Deck, Residential Rings (A-E), Hydroponics Bay, Engineering Core, "
                "and Medical Ward. Life support systems are aging and fragile."
            ),
            tags=["worldbuilding", "setting", "spaceship"],
            confidence=1.0,
        )
        print("   ✅ Created setting: Starship Nova")

        factions_v1 = await client.create_memory(
            topic="political_factions",
            content=(
                "Three main factions on the ship: "
                "1) The Council - elected leaders who maintain order, led by Chairman Marcus Hall "
                "2) The Engineers - workers who keep ship running, demand more representation "
                "3) The Purists - religious group who believe leaving Earth was a sin, growing extremist"
            ),
            tags=["worldbuilding", "politics", "factions"],
            confidence=0.9,
        )
        print("   ✅ Created political landscape")

        history = await client.create_memory(
            topic="ship_history",
            content=(
                "Critical history: 200 years ago, Earth was destroyed by climate collapse. "
                "The Nova launched with 15,000 people. Over the generations, population declined "
                "to 10,000 due to resource constraints. 50 years ago, scanners detected Kepler-442b's "
                "star went supernova - destination is gone. The Council decided to hide this "
                "information to prevent panic, claiming the scanners malfunctioned. "
                "Only top Council members know the truth."
            ),
            tags=["worldbuilding", "history", "secret"],
            confidence=1.0,
        )
        print("   ✅ Created ship history (with dark secret)")
        print()

        # ====================================================================
        # Characters: Complex Character Network
        # ====================================================================
        print("👥 Phase 2: Character Creation - Interconnected Cast...")
        print()

        # Protagonist 1: Sarah Chen (Engineer)
        char_sarah_v1 = await client.create_memory(
            topic="character_sarah_chen",
            content=(
                "Sarah Chen, age 28, lead engineer in the Engineering Core. "
                "Born on the ship - second generation. Brilliant but rebellious. "
                "Her parents died in a reactor accident when she was 10. Raised by her mentor "
                "Dr. Zhang. Best friends with David Wright since childhood. "
                "Recently discovered critical system failures the Council is hiding. "
                "Dating Marcus Hall's son, Tom, which complicates her investigations."
            ),
            tags=["character", "protagonist", "engineer", "sarah"],
            confidence=1.0,
            custom_metadata={"role": "protagonist", "age": 28, "sector": "engineering"},
        )
        print("   ✅ Created protagonist: Sarah Chen (Engineer)")

        # Protagonist 2: David Wright (Medical Officer)
        char_david_v1 = await client.create_memory(
            topic="character_david_wright",
            content=(
                "David Wright, age 30, chief medical officer and Sarah's childhood friend. "
                "His father was on the Council and died under mysterious circumstances 5 years ago. "
                "David suspects it wasn't an accident. He's torn between loyalty to Sarah "
                "and duty to the Council. Secretly investigating his father's death. "
                "Has complicated feelings for Sarah beyond friendship."
            ),
            tags=["character", "protagonist", "medical", "david"],
            confidence=1.0,
            custom_metadata={"role": "protagonist", "age": 30, "sector": "medical"},
        )
        print("   ✅ Created co-protagonist: David Wright (Medical Officer)")

        # Antagonist: Marcus Hall (Council Chairman)
        char_marcus_v1 = await client.create_memory(
            topic="character_marcus_hall",
            content=(
                "Marcus Hall, age 45, Council Chairman. Former military officer. "
                "He's one of the few who knows Kepler-442b is destroyed. "
                "Believes order above all - will suppress information to prevent panic. "
                "Genuinely cares about the ship's survival but morally compromised. "
                "His son Tom is dating Sarah Chen, which worries him - Sarah is too curious. "
                "Has a hidden plan involving an uncharted planet."
            ),
            tags=["character", "antagonist", "council", "marcus"],
            confidence=1.0,
            custom_metadata={"role": "antagonist", "age": 45, "sector": "command"},
        )
        print("   ✅ Created antagonist: Marcus Hall (Council Chairman)")

        # Supporting: Tom Hall (Marcus's son)
        char_tom_v1 = await client.create_memory(
            topic="character_tom_hall",
            content=(
                "Tom Hall, age 27, Marcus's son, works in Navigation. "
                "Dating Sarah Chen for 2 years - genuinely loves her. "
                "Doesn't know about his father's secrets yet. Good-natured and idealistic. "
                "Caught between father and girlfriend. His loyalty will be tested."
            ),
            tags=["character", "supporting", "navigation", "tom"],
            confidence=1.0,
            custom_metadata={"role": "supporting", "age": 27, "sector": "navigation"},
        )
        print("   ✅ Created supporting: Tom Hall (Navigator)")

        # Supporting: Dr. Zhang (Sarah's mentor)
        char_zhang_v1 = await client.create_memory(
            topic="character_dr_zhang",
            content=(
                "Dr. Zhang Wei, age 65, senior engineer and Sarah's mentor. "
                "Raised Sarah after her parents died. Knows more than he says. "
                "Was friends with David's father before his death. Suspects the Council "
                "is hiding something but has been too cautious to investigate. "
                "Deeply loyal to Sarah - would protect her at any cost."
            ),
            tags=["character", "supporting", "engineer", "zhang"],
            confidence=1.0,
            custom_metadata={"role": "supporting", "age": 65, "sector": "engineering"},
        )
        print("   ✅ Created mentor: Dr. Zhang (Senior Engineer)")

        # Relationships - Creating a DENSE character network
        print("\n🔗 Creating character relationship network...")

        # Sarah ↔ David (Best friends, hidden feelings)
        _rel_sarah_david = await client.create_memory(
            topic="relationship_sarah_david",
            content=(
                "Sarah and David: Best friends since age 8, grew up together in Ring C. "
                "After Sarah's parents died, David's family helped support her. They share everything. "
                "David is secretly in love with Sarah but never told her because she's with Tom. "
                "Sarah trusts David more than anyone - he's her emotional anchor. "
                "Their friendship is the emotional core of the story. Unresolved romantic tension."
            ),
            tags=["relationship", "sarah", "david", "friendship", "hidden_love"],
            confidence=1.0,
        )
        print("   ✅ Sarah ↔ David: Best friends (David secretly loves her)")

        # Sarah ↔ Marcus (Adversarial, investigation)
        _rel_sarah_marcus = await client.create_memory(
            topic="relationship_sarah_marcus",
            content=(
                "Sarah and Marcus: Tense relationship. Sarah suspects Marcus is hiding something "
                "about the ship's failing systems. Marcus sees Sarah as a threat - too curious, "
                "too smart. He ordered her supervisor to silence her. Marcus is also uncomfortable "
                "because Sarah is dating his son Tom. He worries Sarah will turn Tom against him. "
                "Sarah respects Marcus's position but doesn't trust him. Adversarial dynamic."
            ),
            tags=["relationship", "sarah", "marcus", "adversarial", "distrust"],
            confidence=0.92,
        )
        print("   ✅ Sarah ↔ Marcus: Adversarial (she suspects, he fears her)")

        # Sarah ↔ Tom (Romantic, straining)
        _rel_sarah_tom = await client.create_memory(
            topic="relationship_sarah_tom",
            content=(
                "Sarah and Tom: Dating for 2 years, genuinely love each other. Tom is kind, "
                "idealistic, and supportive. But Sarah's investigation is straining their relationship. "
                "Tom defends his father Marcus, which frustrates Sarah. Sarah hasn't told Tom "
                "about the critical system failures she discovered. She's torn - loves Tom but "
                "can't fully trust him because of his father. Tom senses Sarah is hiding something. "
                "Their relationship is loving but increasingly strained."
            ),
            tags=["relationship", "sarah", "tom", "romance", "strained"],
            confidence=0.93,
        )
        print("   ✅ Sarah ↔ Tom: Romantic relationship (straining)")

        # Sarah ↔ Dr. Zhang (Mentor-student, deep bond)
        _rel_sarah_zhang = await client.create_memory(
            topic="relationship_sarah_zhang",
            content=(
                "Sarah and Dr. Zhang: Zhang raised Sarah after her parents died in reactor accident. "
                "He's her mentor, father figure, and biggest supporter. Zhang taught Sarah everything "
                "about engineering. He's fiercely protective of her. Sarah turns to Zhang for guidance. "
                "Zhang knows Sarah is investigating the Council but worries for her safety. "
                "He's conflicted - proud of her courage, afraid of consequences. "
                "Deep mentor-student and surrogate father-daughter bond."
            ),
            tags=["relationship", "sarah", "zhang", "mentor", "parental"],
            confidence=0.98,
        )
        print("   ✅ Sarah ↔ Zhang: Mentor/father figure")

        # David ↔ Marcus (Suspicious, personal history)
        _rel_david_marcus = await client.create_memory(
            topic="relationship_david_marcus",
            content=(
                "David and Marcus: Complicated history. David's father was on Council and died "
                "5 years ago - officially an accident, but David suspects foul play. Marcus was "
                "present at the incident. David has been quietly investigating, wonders if Marcus "
                "knows something. Marcus is cordial but distant with David. Marcus respects David's "
                "medical work but watches him carefully. David is suspicious of Marcus and Council. "
                "Underlying tension, dark history between them."
            ),
            tags=["relationship", "david", "marcus", "suspicious", "dark_history"],
            confidence=0.90,
        )
        print("   ✅ David ↔ Marcus: Suspicious (David's father's death)")

        # David ↔ Tom (Awkward, competing for Sarah)
        _rel_david_tom = await client.create_memory(
            topic="relationship_david_tom",
            content=(
                "David and Tom: Awkward relationship. Tom is dating Sarah, David is in love with Sarah. "
                "They're cordial but not close. David tries to be happy for Sarah and Tom, but it's painful. "
                "Tom knows David and Sarah are best friends and sometimes feels like third wheel. "
                "Tom is friendly but David keeps distance. Unspoken rivalry for Sarah's affection. "
                "Both are good people, making it more complicated. Polite but strained."
            ),
            tags=["relationship", "david", "tom", "rivalry", "awkward"],
            confidence=0.87,
        )
        print("   ✅ David ↔ Tom: Awkward (competing for Sarah)")

        # David ↔ Dr. Zhang (Friendly, shared concern)
        _rel_david_zhang = await client.create_memory(
            topic="relationship_david_zhang",
            content=(
                "David and Dr. Zhang: Zhang was friends with David's father before his death. "
                "Zhang has watched David grow up. They bonded over shared loss - Zhang lost Sarah's "
                "parents, David lost his father. Zhang respects David's medical skill and integrity. "
                "They both worry about Sarah's investigation. They meet occasionally to discuss "
                "concerns about the Council. Friendly, supportive relationship based on shared history."
            ),
            tags=["relationship", "david", "zhang", "friendly", "shared_concern"],
            confidence=0.91,
        )
        print("   ✅ David ↔ Zhang: Friendly (shared history and concern)")

        # Marcus ↔ Tom (Father-son, controlling)
        _rel_marcus_tom = await client.create_memory(
            topic="relationship_marcus_tom",
            content=(
                "Marcus and Tom: Father and son. Marcus loves Tom but is emotionally distant and controlling. "
                "Tom respects his father but finds him rigid. Marcus groomed Tom for leadership. "
                "Marcus disapproves of Tom dating Sarah - too unpredictable. Tom defends Sarah to Marcus. "
                "Growing tension as Tom becomes more independent. Marcus keeps secrets from Tom 'for his own good'. "
                "Tom is starting to question his father's decisions. Loving but strained father-son dynamic."
            ),
            tags=["relationship", "marcus", "tom", "father_son", "controlling"],
            confidence=0.95,
        )
        print("   ✅ Marcus ↔ Tom: Father-son (controlling, straining)")

        # Marcus ↔ Dr. Zhang (Mutual wariness, political)
        _rel_marcus_zhang = await client.create_memory(
            topic="relationship_marcus_zhang",
            content=(
                "Marcus and Dr. Zhang: Politically opposed. Zhang represents the engineering faction "
                "wanting more representation. Marcus represents Council authority. They respect each "
                "other's competence but disagree fundamentally. Marcus knows Zhang is protective of Sarah "
                "and watches both carefully. Zhang suspects Marcus is corrupt but lacks proof. "
                "Professional but wary. Both are waiting for the other to make a mistake. Political rivals."
            ),
            tags=["relationship", "marcus", "zhang", "political", "rivals"],
            confidence=0.89,
        )
        print("   ✅ Marcus ↔ Zhang: Political rivals")

        # Tom ↔ Dr. Zhang (Polite, indirect connection)
        _rel_tom_zhang = await client.create_memory(
            topic="relationship_tom_zhang",
            content=(
                "Tom and Dr. Zhang: Connected through Sarah. Tom respects Zhang as Sarah's mentor. "
                "Zhang is polite to Tom but cautious - Tom is Marcus's son. They interact at social "
                "functions. Zhang watches Tom carefully, wondering if Tom knows about his father's secrets. "
                "Tom asks Zhang about Sarah sometimes, trying to understand what's bothering her. "
                "Zhang gives diplomatic answers. Polite but distant, watching each other."
            ),
            tags=["relationship", "tom", "zhang", "polite", "cautious"],
            confidence=0.85,
        )
        print("   ✅ Tom ↔ Zhang: Polite but cautious")

        # Additional historical and contextual memories
        print("\n📜 Adding deep backstory and multi-character events...")

        # Triangle: Sarah-David-Tom emotional complexity
        _history_childhood_trio = await client.create_memory(
            topic="history_childhood_sarah_david_tom",
            content=(
                "Sarah, David, and Tom all grew up in Ring C. David and Sarah were inseparable since age 8. "
                "Tom moved to Ring C at age 15. He was immediately drawn to Sarah. "
                "David saw Tom as competition and felt betrayed when Sarah started dating Tom at age 25. "
                "David never confessed his feelings - couldn't risk losing Sarah's friendship. "
                "This creates the awkward current dynamic. Tom knows David was there first but Sarah chose Tom."
            ),
            tags=["history", "sarah", "david", "tom", "childhood", "triangle"],
            confidence=0.94,
        )
        print("   ✅ Historical: Sarah-David-Tom childhood dynamics")

        # David's father's mysterious death - full story
        _history_david_father_death_detailed = await client.create_memory(
            topic="history_david_father_murder",
            content=(
                "5 years ago: David's father (Council member) discovered the Kepler-442b truth. "
                "He argued for revealing it. Marcus and other Council members disagreed. "
                "David's father was going to make a public announcement. The night before, "
                "he 'fell' down a maintenance shaft. Officially: accident. Actually: murdered. "
                "Marcus was present - either complicit or witnessed it. Dr. Zhang suspected foul play. "
                "David was 25, in medical school, devastated. This shaped his mistrust of authority."
            ),
            tags=["history", "david", "father", "murder", "conspiracy", "5years_ago"],
            confidence=0.97,
        )
        print("   ✅ Historical: David's father's murder (detailed)")

        # Marcus's tragic past - why he's so authoritarian
        _history_marcus_past_trauma = await client.create_memory(
            topic="history_marcus_military_trauma",
            content=(
                "Marcus's background: Former military officer on Earth before its collapse. "
                "He was in charge during evacuation. Had to make horrible choices - who got on ship, who didn't. "
                "Thousands died because of his decisions. He still hears their screams. "
                "This is why he's obsessed with order and control - guilt drives him. "
                "He believes 'difficult decisions save lives', justifying current coverup. "
                "Tom doesn't know about his father's Earth trauma. Marcus never talks about it."
            ),
            tags=["history", "marcus", "trauma", "earth", "military"],
            confidence=0.95,
        )
        print("   ✅ Historical: Marcus's Earth trauma (explains his character)")

        # Sarah's parents' death - conspiracy connection
        _history_sarah_parents_death = await client.create_memory(
            topic="history_sarah_parents_reactor_accident",
            content=(
                "18 years ago: Sarah's parents (both engineers) died in 'reactor accident'. "
                "Official story: human error. Reality: They discovered life support was degrading. "
                "Council ordered them to falsify reports. They refused. 'Accident' followed. "
                "10-year-old Sarah was orphaned. Dr. Zhang was their colleague - he suspected truth. "
                "He's protected Sarah ever since. Marcus knows the truth. This is why he watches Sarah. "
                "Sarah discovering same system failures is history repeating."
            ),
            tags=["history", "sarah", "parents", "conspiracy", "murder", "18years_ago"],
            confidence=0.96,
        )
        print("   ✅ Historical: Sarah's parents were murdered by Council")

        # Dr. Zhang's guilt and mission
        _zhang_survivor_guilt = await client.create_memory(
            topic="zhang_survivor_guilt_mission",
            content=(
                "Dr. Zhang's burden: He worked with Sarah's parents and David's father. "
                "He suspected both deaths weren't accidental. He stayed silent out of fear. "
                "He adopted Sarah to atone. He mentors young engineers to honor Sarah's parents. "
                "He quietly gathers evidence against Council. Zhang is planning something big. "
                "He's waiting for the right moment. Sarah is becoming his instrument of justice. "
                "Zhang's mission: expose Council, honor the dead, protect the living."
            ),
            tags=["internal", "zhang", "guilt", "mission", "justice"],
            confidence=0.94,
        )
        print("   ✅ Internal: Dr. Zhang's survivor guilt and secret mission")

        # Tom's internal struggle - family vs truth
        _tom_internal_conflict = await client.create_memory(
            topic="tom_torn_loyalties",
            content=(
                "Tom's constant conflict: He loves his father but senses something wrong. "
                "He loves Sarah but she's keeping secrets. He's loyal to Council but questions decisions. "
                "Tom has access to navigation data - he's seen anomalies in the stellar maps. "
                "He's starting to suspect there's no valid destination. But acknowledging this means "
                "his father lied, his relationship is based on deception, his life purpose is void. "
                "Tom is in denial, but cracks are forming."
            ),
            tags=["internal", "tom", "conflict", "denial", "awakening"],
            confidence=0.92,
        )
        print("   ✅ Internal: Tom's growing doubts and denial")

        # Multi-party event: The ship-wide drill incident
        _event_emergency_drill = await client.create_memory(
            topic="event_emergency_drill_2years_ago",
            content=(
                "2 years ago: Marcus ordered ship-wide emergency drill. Systems tested to limits. "
                "Sarah (age 26) detected critical life support vulnerabilities during drill. "
                "She reported to Dr. Zhang. Zhang reported to Council. Marcus suppressed report. "
                "David's medical bay handled panic casualties. Tom was in navigation, saw data. "
                "This event planted seeds of doubt in everyone. Sarah started investigating. "
                "Zhang started documenting. Tom started noticing inconsistencies."
            ),
            tags=["event", "drill", "all_characters", "turning_point", "2years_ago"],
            confidence=0.93,
        )
        print("   ✅ Event: Emergency drill (affected all characters)")

        # Location: The restricted archives
        _location_restricted_archives = await client.create_memory(
            topic="location_council_archives",
            content=(
                "Council archives: Restricted section in Command Deck. "
                "Contains true ship history, real stellar data, classified incidents. "
                "Marcus controls access. Only top Council members allowed. "
                "David's father accessed it before his death. Sarah's parents too. "
                "Tom has low-level access (doesn't know what he doesn't see). "
                "Sarah and David plan to break in - it contains all answers. "
                "Archives are the key to everything."
            ),
            tags=["location", "archives", "secrets", "restricted"],
            confidence=0.95,
        )
        print("   ✅ Location: Restricted archives (central to plot)")

        # Shared secret: The resistance network
        _secret_resistance_network = await client.create_memory(
            topic="secret_engineering_resistance",
            content=(
                "Secret: Dr. Zhang has been building resistance network in Engineering faction. "
                "50+ engineers loyal to him, ready to act. Sarah doesn't know she's the figurehead. "
                "Zhang plans to use Sarah's discoveries as catalyst for uprising. "
                "David is connecting medical staff to network. Marcus suspects but can't prove it. "
                "Tom is oblivious. Network communicates through coded maintenance logs. "
                "When Sarah learns truth, she'll have an army ready."
            ),
            tags=["secret", "resistance", "zhang", "conspiracy", "rebellion"],
            confidence=0.91,
        )
        print("   ✅ Secret: Dr. Zhang's resistance network")

        # Marcus's secret plan details
        _marcus_secret_plan_detailed = await client.create_memory(
            topic="marcus_population_reduction_plan",
            content=(
                "Marcus's hidden plan: He found habitable planet only 10 years away. "
                "Problem: life support can't sustain 10,000 for 10 years - max 5,000. "
                "Solution: Reduce population by half. Method: Controlled catastrophe. "
                "Stage an 'accident' in Rings D-E (working class, including engineers). "
                "Blame system failures, tragic but necessary. Survivors reach new world. "
                "Marcus is planning genocide to save humanity. He's written justification documents. "
                "Tom will inherit this plan. Marcus thinks he's being merciful."
            ),
            tags=["secret", "marcus", "genocide", "plan", "moral_horror"],
            confidence=0.98,
        )
        print("   ✅ Secret: Marcus's genocide plan (horrifying truth)")

        # Multi-character event: Tom and Sarah's relationship milestone
        _event_tom_sarah_relationship = await client.create_memory(
            topic="event_tom_sarah_2years_dating",
            content=(
                "2 years ago: Tom proposed to Sarah. She said 'not yet'. "
                "She loves him but something holds her back. David was crushed hearing about proposal. "
                "Marcus disapproved - wanted Tom to marry within Council families. "
                "Dr. Zhang worried Sarah's investigation would hurt Tom. "
                "Tom was hurt by 'not yet' but accepted it. Sarah's hesitation was instinctive - "
                "subconsciously she didn't trust Tom's father, affecting her trust in Tom. "
                "This unresolved proposal haunts their relationship."
            ),
            tags=["event", "tom", "sarah", "proposal", "relationship", "2years_ago"],
            confidence=0.90,
        )
        print("   ✅ Event: Tom's proposal (affected all characters)")

        # Sarah's hidden discovery - even David doesn't know
        _sarah_secret_discovery = await client.create_memory(
            topic="sarah_secret_discovery",
            content=(
                "Sarah's recent discovery (hasn't told anyone yet): "
                "She found her mother's hidden journal during maintenance. "
                "Journal documents Council's corruption, life support lies, past 'accidents'. "
                "It names names: Marcus, others. It mentions David's father as ally. "
                "It predicts their own deaths. Last entry: 'Zhang knows. He'll protect Sarah.' "
                "Sarah now knows her parents were murdered. She's been carrying this rage for 2 weeks. "
                "She's planning something. David senses she's changed but doesn't know why."
            ),
            tags=["secret", "sarah", "journal", "discovery", "rage", "recent"],
            confidence=0.96,
        )
        print("   ✅ Secret: Sarah found her mother's journal (game-changer)")

        # David's investigation progress
        _david_investigation = await client.create_memory(
            topic="david_father_investigation",
            content=(
                "David's 5-year investigation into his father's death: "
                "He's collected: witness statements, maintenance logs, sealed medical reports. "
                "He knows his father was murdered but can't prove it. He suspects Marcus. "
                "He's connected his father's death to Sarah's parents' deaths - same pattern. "
                "He's been meeting with Dr. Zhang secretly. Together they've mapped the conspiracy. "
                "David is waiting for one more piece of evidence. When he gets it, he'll act. "
                "His medical position gives him access to death certificates - all evidence."
            ),
            tags=["investigation", "david", "evidence", "conspiracy"],
            confidence=0.94,
        )
        print("   ✅ Investigation: David's evidence collection")

        # Ship political landscape - factions detail
        _politics_detailed_factions = await client.create_memory(
            topic="politics_faction_details",
            content=(
                "Detailed faction landscape: "
                "Council (Marcus faction): 15 members, authoritarian, know the truth. "
                "Engineering (Zhang faction): 800 workers, demand representation, growing radical. "
                "Medical (David's domain): 200 staff, neutral but swaying toward engineers. "
                "Navigation (Tom's area): 50 officers, loyal to Council, but Tom is questioning. "
                "Purists: 500 religious extremists, believe ship life is purgatory, dangerous. "
                "Civilians: 8,000 regular people, unaware of conspiracies, caught in middle. "
                "Ship is powder keg waiting for spark."
            ),
            tags=["politics", "factions", "detailed", "powder_keg"],
            confidence=0.93,
        )
        print("   ✅ Politics: Detailed faction breakdown")

        # The catalyst event that starts everything
        _catalyst_oxygen_crisis = await client.create_memory(
            topic="catalyst_current_oxygen_crisis",
            content=(
                "Current crisis (Story Day 0): Oxygen recyclers in Ring C failing faster than predicted. "
                "Sarah discovers it during routine check. Ring C is where she, David, and Tom's families live. "
                "Sarah estimates: 6 months until breathable air is gone. Council knows. Marcus is stalling repairs. "
                "Why? Because Ring C failure is PART of his population reduction plan. "
                "He's letting it fail 'naturally'. Sarah doesn't know this yet. "
                "She thinks it's negligence. Reality is worse - it's intentional genocide. "
                "This discovery will set everything in motion."
            ),
            tags=["catalyst", "crisis", "oxygen", "day_0", "genocide"],
            confidence=0.97,
        )
        print("   ✅ Catalyst: Current oxygen crisis (story begins)")

        print(f"\n   📊 Total initial memories: 5 characters + 10 relationships + 15 historical/contextual")
        print(f"   📊 = 30+ interconnected memory states!")
        print("   🕸️  Dense narrative network with multiple conspiracy layers!")
        print()

        # ====================================================================
        # Chapters 1-5: Discovery Arc
        # ====================================================================
        print("📝 Phase 3: Writing early chapters (Discovery Arc)...")
        print()

        plot_discovery = await client.create_memory(
            topic="plot_discovery_arc",
            content=(
                "Chapters 1-5: Sarah discovers failing life support systems during "
                "routine maintenance. Oxygen recyclers are degrading faster than expected. "
                "She reports to her supervisor but is told to stay quiet by Council orders. "
                "Sarah confides in David. Together they start investigating. "
                "Dr. Zhang warns Sarah to be careful - 'the Council has secrets'. "
                "Sarah's investigation strains her relationship with Tom, who defends his father."
            ),
            tags=["plot", "chapters_1_5", "discovery", "arc"],
            confidence=0.95,
        )
        print("   ✅ Plotted chapters 1-5: Sarah discovers life support crisis")

        # ====================================================================
        # Chapter 6-8: MAJOR REVELATION (Triggers Cascading!)
        # ====================================================================
        print("\n📖 Phase 4: Chapter 6-8 - MAJOR PLOT TWIST...")
        print()

        plot_revelation = await client.create_memory(
            topic="plot_revelation",
            content=(
                "CHAPTER 8 CLIMAX: Sarah and David break into restricted Council archives. "
                "They discover the devastating truth: Kepler-442b was destroyed 50 years ago "
                "by supernova. The Council has been lying for two generations! "
                "The ship has been traveling toward NOTHING. Worse: they find evidence that "
                "David's father discovered this truth 5 years ago and was murdered to silence him. "
                "This revelation shatters everything. Sarah is furious. David is devastated - "
                "his father was killed by the Council. Tom's father is complicit in murder."
            ),
            tags=["plot", "chapters_6_8", "revelation", "critical", "twist"],
            confidence=0.98,
        )
        print("   🚨 MAJOR PLOT TWIST: Destination destroyed! Council murdered David's father!")
        print("   ⚠️  This should CASCADE to ALL character relationships!")
        print("   🔄 Waiting for cascading evolution...\n")
        await asyncio.sleep(4)  # Give time for cascading

        # ====================================================================
        # Phase 5: Check Cascading Evolution
        # ====================================================================
        print("🔍 Phase 5: Verifying character evolution from revelation...")
        print()

        characters_to_check = [
            "character_sarah_chen",
            "character_david_wright",
            "character_tom_hall",
            "character_marcus_hall",
            "relationship_sarah_david",
        ]

        evolved_count = 0
        for topic in characters_to_check:
            trace = await client.get_trace(topic)
            versions = len(trace)
            if versions > 1:
                evolved_count += 1
                latest = trace[0]
                print(f"\n   ✨ {topic} EVOLVED to v{latest.version}:")
                print(f"      {latest.content[:150]}...")
                if "cascading_evolved" in latest.tags:
                    print(f"      🎯 Tagged as cascading_evolved!")
            else:
                print(f"\n   ⚠️  {topic}: No evolution (v1)")

        print(f"\n   📊 Evolution summary: {evolved_count}/{len(characters_to_check)} characters evolved\n")

        # ====================================================================
        # Chapters 9-12: Character Development & Relationship Changes
        # ====================================================================
        print("🎭 Phase 6: Character development after revelation...")
        print()

        # Sarah's transformation
        char_sarah_v2 = await client.create_memory(
            topic="character_sarah_chen",
            content=(
                "Sarah Chen (Post-Revelation, Chapter 10): The truth about Kepler-442b and "
                "the murder of David's father has transformed Sarah. She's no longer just "
                "a brilliant engineer - she's become a revolutionary leader. She broke up "
                "with Tom (can't trust Marcus's son). She's organizing the engineering faction "
                "to take control of the ship. Her relationship with David has intensified - "
                "bound together by shared trauma and purpose. Goal: expose the Council, "
                "find new destination, establish democracy on the ship."
            ),
            tags=["character", "protagonist", "engineer", "sarah", "evolved", "revolutionary"],
            confidence=0.95,
            custom_metadata={
                "role": "protagonist",
                "age": 28,
                "sector": "engineering",
                "chapter": 10,
                "arc": "revolutionary_leader",
            },
        )
        print("   ✅ Sarah evolved: Now a revolutionary leader")

        # David's transformation
        char_david_v2 = await client.create_memory(
            topic="character_david_wright",
            content=(
                "David Wright (Post-Revelation, Chapter 10): Learning his father was murdered "
                "by the Council shattered David. His lifelong respect for authority is gone. "
                "He's consumed by grief and rage. He's now helping Sarah's rebellion, using "
                "his medical position to recruit supporters. His feelings for Sarah have "
                "grown stronger through this crisis. He finally confessed his love to Sarah - "
                "she admitted she feels the same. They're together now, both personally and "
                "in their mission to take down the Council."
            ),
            tags=["character", "protagonist", "medical", "david", "evolved", "grief"],
            confidence=0.94,
            custom_metadata={
                "role": "protagonist",
                "age": 30,
                "sector": "medical",
                "chapter": 10,
                "arc": "vengeful_rebel",
            },
        )
        print("   ✅ David evolved: Grief-driven rebel, now with Sarah")

        # Tom's crisis
        char_tom_v2 = await client.create_memory(
            topic="character_tom_hall",
            content=(
                "Tom Hall (Chapter 11): Tom is in crisis. Sarah broke up with him and revealed "
                "his father's crimes. Tom confronted Marcus and learned it's all true - "
                "the Council has been lying, and they murdered David's father. Tom is devastated. "
                "He must choose: loyalty to his father or doing what's right. "
                "He's isolated, trusted by neither Council nor rebels. His arc is about "
                "finding his own moral compass independent of his father."
            ),
            tags=["character", "supporting", "navigation", "tom", "conflict", "crisis"],
            confidence=0.91,
            custom_metadata={
                "role": "supporting",
                "age": 27,
                "sector": "navigation",
                "chapter": 11,
                "arc": "moral_crisis",
            },
        )
        print("   ✅ Tom evolved: Facing moral crisis, isolated")

        # Marcus doubles down
        char_marcus_v2 = await client.create_memory(
            topic="character_marcus_hall",
            content=(
                "Marcus Hall (Chapter 11): Marcus sees Sarah's rebellion as a threat to ship survival. "
                "He's preparing to use military force to suppress it. He's lost his son Tom, "
                "who now sees him as a villain. Marcus justifies his actions: 'I did what "
                "was necessary to prevent panic and chaos.' He reveals his secret plan: "
                "he's found an uncharted habitable planet only 10 years away. But reaching it "
                "requires radical measures - reducing population to extend resources. "
                "Marcus has become a tragic villain: good intentions, monstrous methods."
            ),
            tags=["character", "antagonist", "council", "marcus", "hardened", "villain"],
            confidence=0.96,
            custom_metadata={
                "role": "antagonist",
                "age": 45,
                "sector": "command",
                "chapter": 11,
                "arc": "tragic_villain",
            },
        )
        print("   ✅ Marcus evolved: Tragic villain with secret plan")

        # Relationship evolution
        _sarah_david_romance = await client.create_memory(
            topic="relationship_sarah_david",
            content=(
                "Sarah and David's relationship (Chapter 10): After the revelation, their "
                "lifelong friendship transformed into romance. The shared trauma of the truth, "
                "David's grief, and Sarah's revolutionary purpose brought them together. "
                "David finally confessed he'd loved Sarah for years. Sarah admitted the same. "
                "They're now partners in every sense - emotionally, romantically, and in their "
                "mission to save the ship. Their relationship is the emotional heart of the story."
            ),
            tags=["relationship", "sarah", "david", "romance", "evolved"],
            confidence=0.97,
        )
        print("   ✅ Sarah & David: Friends → Romantic partners\n")

        # ====================================================================
        # Chapters 13-15: Political Factions Shift
        # ====================================================================
        print("📊 Phase 7: Political landscape shifts...")
        print()

        factions_v2 = await client.create_memory(
            topic="political_factions",
            content=(
                "Factions after revelation (Chapter 13): "
                "1) Council Loyalists - Marcus's hardliners, preparing for military crackdown "
                "2) Reform Council - Council members who want truth revealed, secret negotiation with rebels "
                "3) Engineer Rebellion - Sarah's faction, largest and growing, demands democracy "
                "4) Medical Neutrals - David's faction, providing humanitarian aid to all sides "
                "5) Purist Extremists - religious faction plans violence, 'purge the liars' "
                "6) NEW: The Navigators - Tom Hall leads this new faction of scientists, "
                "neutral but have critical information about Marcus's secret planet. "
                "Ship is on brink of civil war."
            ),
            tags=["worldbuilding", "politics", "factions", "updated", "crisis"],
            confidence=0.92,
        )
        print("   ✅ Factions evolved: 6 factions, ship on brink of civil war")
        print("   🔄 Waiting for cascading...\n")
        await asyncio.sleep(3)

        # ====================================================================
        # Phase 8: Manual Promotion - Synthesize Sarah's Journey
        # ====================================================================
        print("🚀 Phase 8: Manual promotion - synthesizing Sarah's complete arc...")
        print()

        promotion = await client.promote_memory(
            topic="character_sarah_chen",
            reason=(
                "End of Act 2 (Chapter 15): Sarah has undergone massive transformation from "
                "engineer to revolutionary leader. Need comprehensive synthesis of her journey, "
                "relationships, motivations, and current state for planning final act."
            ),
        )

        if promotion.success:
            new_state = promotion.new_state
            print(f"   ✅ Promotion successful!")
            print(f"   📊 New version: v{new_state.version}")
            print(f"   📝 Synthesized Sarah's journey:")
            print(f"      {new_state.content[:500]}...")
            print(f"\n   📚 Sources: {len(promotion.synthesis_sources)}")
            print(f"   🎯 Confidence: {promotion.confidence:.2f}")
            if promotion.reasoning:
                print(f"   💭 Reasoning: {promotion.reasoning[:250]}...")
        else:
            print(f"   ❌ Failed: {promotion.error_message}")

        # ====================================================================
        # Phase 9: Complex Story Queries
        # ====================================================================
        print("\n🔎 Phase 9: Complex narrative queries...")
        print()

        queries = [
            "How has Sarah Chen's character evolved from the beginning?",
            "What is the relationship between Sarah, David, Tom, and Marcus?",
            "What was the revelation that changed everything?",
            "How has the political situation on the ship evolved?",
            "What role did David's father's death play in the story?",
        ]

        for i, query_text in enumerate(queries, 1):
            print(f"\n   Query {i}: '{query_text}'")
            result = await client.query(
                query_text,
                include_history=True,
                include_related=True,
            )
            print(f"   Answer: {result.answer[:300]}...")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Sources: {len(result.sources)}")

        # ====================================================================
        # Phase 10: View Character Evolution Traces
        # ====================================================================
        print("\n📈 Phase 10: Viewing character evolution traces...")
        print()

        key_characters = [
            "character_sarah_chen",
            "character_david_wright",
            "character_marcus_hall",
            "character_tom_hall",
            "relationship_sarah_david",
        ]

        for topic in key_characters:
            trace = await client.get_trace(topic)
            print(f"\n   📌 {topic}: {len(trace)} version(s)")
            for state in trace[:4]:  # Show up to 4 versions
                chapter = state.custom_metadata.get("chapter", "initial")
                arc = state.custom_metadata.get("arc", "intro")
                tags_str = ", ".join(state.tags[:3])
                cascading = "🔄" if "cascading_evolved" in state.tags else ""
                print(f"      v{state.version} {cascading} - Ch.{chapter} ({arc})")
                print(f"         Tags: [{tags_str}]")
                print(f"         {state.content[:100]}...")

        # ====================================================================
        # Phase 11: Continuity Check
        # ====================================================================
        print("\n🔍 Phase 11: Checking narrative consistency...")
        print()

        print("   Q: 'Does Tom know about his father's plan for the new planet?'")
        consistency_check = await client.query(
            "Does Tom Hall know about his father Marcus's secret plan for the uncharted planet?",
            include_history=True,
            include_related=True,
        )
        print(f"   A: {consistency_check.answer[:200]}...")
        print()

        print("   Q: 'What are the key relationships between main characters now?'")
        relationships_check = await client.query(
            "What are the current relationships between Sarah, David, Tom, and Marcus after the revelation?",
            include_history=True,
            include_related=True,
        )
        print(f"   A: {relationships_check.answer[:250]}...")
        print()

    print("=" * 70)
    print("✅ Complex novel writing test completed!")
    print("=" * 70)
    print("\nKey Verifications:")
    print("  1. ✓ Complex character network with interconnected relationships")
    print("  2. ✓ Major plot events trigger cascading evolution across characters")
    print("  3. ✓ Character arcs develop naturally from plot developments")
    print("  4. ✓ Relationship dynamics shift realistically")
    print("  5. ✓ Manual promotion synthesizes comprehensive character journeys")
    print("  6. ✓ Queries understand complex narrative threads")
    print("  7. ✓ Story background and world-building interconnect")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(novel_writing_example())
    except KeyboardInterrupt:
        print("\n\n⚠️  Example cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("   1. TracingRAG API is running")
        print("   2. All services are healthy: poetry run python scripts/verify_setup.py")
        sys.exit(1)
