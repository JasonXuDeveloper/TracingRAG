#!/usr/bin/env python3
"""
Example 2: Complex NPC Memory & Multi-Character Simulation

This ENHANCED example demonstrates advanced features:
- Multiple NPCs interacting with EACH OTHER (not just player)
- Cascading evolution when one NPC's state affects others
- Complex relationship networks and social dynamics
- Manual promotion for synthesizing group dynamics
- Relationship manager tracking inter-NPC connections

Use Case: Video games with complex NPC ecosystems, simulations, social dynamics
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tracingrag.client import AsyncTracingRAGClient


async def simulate_npc_interactions():
    """Simulate complex NPC ecosystem with multiple relationships"""
    print("=" * 70)
    print("🎭 COMPLEX Test: Multi-NPC Social Simulation")
    print("=" * 70)
    print()

    # Initialize client
    print("📡 Connecting to TracingRAG API...")
    async with AsyncTracingRAGClient("http://localhost:8000", timeout=120.0) as client:
        health = await client.health()
        if health["status"] != "healthy":
            print("❌ API is not healthy. Please start the API server.")
            return
        print("✅ Connected to TracingRAG API\n")

        # ====================================================================
        # Setup: Define NPCs and their EXISTING relationships
        # ====================================================================
        print("🎭 Phase 1: Setting up NPC ecosystem...")
        print()

        # NPC 1: Merchant Elena (hub of social network)
        _elena_intro = await client.create_memory(
            topic="npc_elena_merchant",
            content=(
                "Elena is a kind merchant who runs the general store in Rivertown. "
                "She's been trading for 20 years and knows everyone in town. "
                "She values honesty and fair dealing. Her husband was a blacksmith "
                "who passed away 5 years ago. She's the town's unofficial gossip hub - "
                "everyone comes to her shop, so she hears everything. "
                "She's close friends with Thomas the innkeeper and often worries about Marcus."
            ),
            tags=["npc", "merchant", "elena", "social_hub"],
            confidence=1.0,
            custom_metadata={"npc_type": "merchant", "location": "rivertown_store"},
        )
        print("   ✅ Created NPC: Elena (Merchant & Social Hub)")

        # NPC 2: Guard Captain Marcus (authority figure)
        _marcus_intro = await client.create_memory(
            topic="npc_marcus_guard",
            content=(
                "Marcus is the stern but fair captain of the town guard. "
                "He's suspicious of strangers and takes his duty seriously. "
                "He's been protecting Rivertown for 10 years and has seen many threats. "
                "Marcus has a secret: he's in love with Elena but hasn't told her. "
                "He often visits her shop just to talk. He trusts Thomas but is "
                "suspicious of Silas, the newcomer merchant."
            ),
            tags=["npc", "guard", "marcus", "authority"],
            confidence=1.0,
            custom_metadata={"npc_type": "guard_captain", "location": "guard_barracks"},
        )
        print("   ✅ Created NPC: Marcus (Guard Captain)")

        # NPC 3: Innkeeper Thomas (peacemaker)
        _thomas_intro = await client.create_memory(
            topic="npc_thomas_innkeeper",
            content=(
                "Thomas runs the Riverside Inn, the town's main gathering place. "
                "Age 55, he's a jovial man who knows everyone and keeps the peace. "
                "He's Elena's oldest friend - they grew up together. "
                "Thomas is the mediator when disputes arise. He knows about Marcus's "
                "feelings for Elena and tries to encourage him. "
                "He's also the one who hired Lyra, the young barmaid."
            ),
            tags=["npc", "innkeeper", "thomas", "peacemaker"],
            confidence=1.0,
            custom_metadata={"npc_type": "innkeeper", "location": "riverside_inn"},
        )
        print("   ✅ Created NPC: Thomas (Innkeeper & Mediator)")

        # NPC 4: Merchant Silas (rival & troublemaker)
        _silas_intro = await client.create_memory(
            topic="npc_silas_merchant",
            content=(
                "Silas is a new merchant who arrived 6 months ago. He opened a competing "
                "store across from Elena's. Age 40, charming but shrewd. "
                "He undercuts Elena's prices and spreads rumors about her. "
                "Marcus suspects Silas has criminal connections. Thomas tries to stay "
                "neutral but doesn't trust Silas. Elena sees him as a business threat."
            ),
            tags=["npc", "merchant", "silas", "rival"],
            confidence=1.0,
            custom_metadata={"npc_type": "merchant", "location": "silas_emporium"},
        )
        print("   ✅ Created NPC: Silas (Rival Merchant)")

        # NPC 5: Barmaid Lyra (catalyst character)
        _lyra_intro = await client.create_memory(
            topic="npc_lyra_barmaid",
            content=(
                "Lyra is a young barmaid (age 22) at Thomas's inn. "
                "She's observant and picks up on things others miss. "
                "She befriended Elena quickly and often helps her at the store. "
                "She has a crush on one of Marcus's guards but is too shy to say anything. "
                "Recently, she noticed something suspicious about Silas's late-night meetings."
            ),
            tags=["npc", "barmaid", "lyra", "observer"],
            confidence=1.0,
            custom_metadata={"npc_type": "barmaid", "location": "riverside_inn"},
        )
        print("   ✅ Created NPC: Lyra (Barmaid & Observer)")

        # Initial relationships - Creating a DENSE social network
        print("\n🔗 Creating relationship network...")

        # Elena ↔ Thomas (Best Friends)
        _rel_elena_thomas = await client.create_memory(
            topic="relationship_elena_thomas",
            content=(
                "Elena and Thomas: Best friends since childhood (40+ years). "
                "They trust each other completely. Thomas gives Elena emotional support "
                "after her husband's death. Elena helps Thomas with inn supplies at cost. "
                "Their friendship is the bedrock of Rivertown's social fabric. "
                "They know each other's secrets."
            ),
            tags=["relationship", "elena", "thomas", "friendship", "deep_bond"],
            confidence=1.0,
        )
        print("   ✅ Elena ↔ Thomas: Childhood best friends")

        # Elena ↔ Marcus (Mutual respect, unspoken feelings)
        _rel_elena_marcus = await client.create_memory(
            topic="relationship_elena_marcus",
            content=(
                "Elena and Marcus: Marcus has been in love with Elena for 3 years but hasn't "
                "confessed. He visits her shop frequently, ostensibly for supplies. "
                "Elena finds Marcus reliable but distant. She respects his dedication to the town. "
                "After her husband's death, Marcus has been extra protective, which Elena notices. "
                "There's romantic tension neither acknowledges."
            ),
            tags=["relationship", "elena", "marcus", "romantic_tension", "unspoken"],
            confidence=0.95,
        )
        print("   ✅ Elena ↔ Marcus: Unspoken romantic tension")

        # Elena ↔ Silas (Business rivals, distrust)
        _rel_elena_silas = await client.create_memory(
            topic="relationship_elena_silas",
            content=(
                "Elena and Silas: Direct business competitors. Silas opened his store 6 months ago "
                "and has been undercutting Elena's prices. Elena dislikes his aggressive tactics "
                "but tries to stay civil. Silas sees Elena as an obstacle to his success. "
                "He spreads subtle rumors about her goods. Elena suspects he's dishonest "
                "but has no proof yet. Cold war between them."
            ),
            tags=["relationship", "elena", "silas", "rivalry", "distrust"],
            confidence=0.90,
        )
        print("   ✅ Elena ↔ Silas: Business rivals, mutual distrust")

        # Elena ↔ Lyra (Mentor-like, sisterly)
        _rel_elena_lyra = await client.create_memory(
            topic="relationship_elena_lyra",
            content=(
                "Elena and Lyra: Lyra reminds Elena of her younger self. They became close quickly. "
                "Lyra helps at Elena's store on her days off. Elena gives Lyra advice about life "
                "and relationships. Elena is protective of Lyra like an older sister. "
                "Lyra confides in Elena about things she notices at the inn. "
                "Strong mentor-mentee and sisterly bond."
            ),
            tags=["relationship", "elena", "lyra", "mentor", "sisterly"],
            confidence=0.92,
        )
        print("   ✅ Elena ↔ Lyra: Mentor/sisterly bond")

        # Marcus ↔ Thomas (Old friends, confidants)
        _rel_marcus_thomas = await client.create_memory(
            topic="relationship_marcus_thomas",
            content=(
                "Marcus and Thomas: Friends for 15+ years. They drink together at the inn weekly. "
                "Thomas is one of the few people Marcus opens up to. Thomas knows about Marcus's "
                "feelings for Elena and encourages him to confess. Marcus trusts Thomas's judgment "
                "on town matters. Thomas acts as mediator when Marcus's strictness causes friction. "
                "Solid, loyal friendship."
            ),
            tags=["relationship", "marcus", "thomas", "friendship", "confidants"],
            confidence=0.95,
        )
        print("   ✅ Marcus ↔ Thomas: Old friends and confidants")

        # Marcus ↔ Silas (Deep suspicion, surveillance)
        _rel_marcus_silas = await client.create_memory(
            topic="relationship_marcus_silas",
            content=(
                "Marcus and Silas: Marcus has suspected Silas since he arrived. Silas's story "
                "doesn't add up - too much money, too few legitimate sources. Marcus has been "
                "quietly investigating Silas for 2 months. Silas knows Marcus is watching him "
                "and is careful. Cold relationship. Marcus interviews Silas monthly, officially "
                "'routine checks'. Silas acts charming but Marcus sees through it. "
                "Cat-and-mouse dynamic."
            ),
            tags=["relationship", "marcus", "silas", "suspicion", "investigation"],
            confidence=0.93,
        )
        print("   ✅ Marcus ↔ Silas: Suspicion and surveillance")

        # Marcus ↔ Lyra (Protective, watchful)
        _rel_marcus_lyra = await client.create_memory(
            topic="relationship_marcus_lyra",
            content=(
                "Marcus and Lyra: Marcus sees Lyra as a daughter figure. Her father was one of "
                "Marcus's guards who died in bandit raid 5 years ago. Marcus helped Lyra get the "
                "barmaid job with Thomas. He checks on her welfare regularly. Lyra respects Marcus "
                "but is a bit intimidated by his stern demeanor. Marcus trusts her honesty - "
                "she once reported a theft she witnessed. Protective guardian-ward dynamic."
            ),
            tags=["relationship", "marcus", "lyra", "protective", "guardian"],
            confidence=0.90,
        )
        print("   ✅ Marcus ↔ Lyra: Protective guardian dynamic")

        # Thomas ↔ Silas (Forced civility, distrust)
        _rel_thomas_silas = await client.create_memory(
            topic="relationship_thomas_silas",
            content=(
                "Thomas and Silas: Silas drinks at Thomas's inn regularly. Thomas serves him "
                "but doesn't trust him. Thomas has noticed Silas meeting shady characters in "
                "back rooms. As innkeeper, Thomas maintains neutrality, but personally he agrees "
                "with Marcus's suspicions. Silas tries to befriend Thomas to gain information "
                "about town affairs. Thomas gives polite but vague answers. Professional but cold."
            ),
            tags=["relationship", "thomas", "silas", "distrust", "forced_civility"],
            confidence=0.88,
        )
        print("   ✅ Thomas ↔ Silas: Forced civility, hidden distrust")

        # Thomas ↔ Lyra (Employer-employee, fatherly)
        _rel_thomas_lyra = await client.create_memory(
            topic="relationship_thomas_lyra",
            content=(
                "Thomas and Lyra: Thomas hired Lyra when she needed work. He treats her like "
                "his own daughter (he has no children). Lyra works hard and Thomas pays her well. "
                "He teaches her about running a business. Lyra trusts Thomas completely and "
                "reports everything unusual she observes. Thomas is proud of her perceptiveness. "
                "Warm employer-employee relationship with fatherly care."
            ),
            tags=["relationship", "thomas", "lyra", "fatherly", "employer"],
            confidence=0.95,
        )
        print("   ✅ Thomas ↔ Lyra: Fatherly employer relationship")

        # Silas ↔ Lyra (Silas unaware, Lyra suspicious)
        _rel_silas_lyra = await client.create_memory(
            topic="relationship_silas_lyra",
            content=(
                "Silas and Lyra: Silas barely notices Lyra - just another barmaid to him. "
                "He's polite but dismissive. CRITICAL: Lyra has been watching Silas closely "
                "for weeks. She's seen him meet with rough-looking men late at night. "
                "She's overheard fragments of conversations about 'shipments' and 'eastern road'. "
                "Lyra is afraid of Silas but hasn't told anyone yet except vague hints to Thomas. "
                "Asymmetric relationship - he's unaware she's observing him."
            ),
            tags=["relationship", "silas", "lyra", "suspicious", "asymmetric"],
            confidence=0.92,
        )
        print("   ✅ Silas ↔ Lyra: Asymmetric (he's unaware, she's watching)")

        # Additional past history memories (triangular relationships)
        print("\n📜 Adding historical events and multi-party relationships...")

        # Triangle: Elena-Marcus-Thomas past event
        _history_elena_marcus_thomas = await client.create_memory(
            topic="history_three_friends",
            content=(
                "15 years ago: Elena, Marcus, and Thomas were all single and often spent evenings together. "
                "There was a love triangle - Marcus loved Elena, but Elena briefly dated Thomas. "
                "When Elena married the blacksmith instead, it hurt both men. Thomas stepped back gracefully. "
                "Marcus was devastated but hid it. The three remained friends but the dynamic changed forever. "
                "This is why Marcus never confessed again, and why Thomas encourages him now."
            ),
            tags=["history", "elena", "marcus", "thomas", "love_triangle", "15years_ago"],
            confidence=0.95,
        )
        print("   ✅ Historical: Elena-Marcus-Thomas love triangle (15 years ago)")

        # Triangle: Marcus-Lyra-Thomas family connection
        _history_lyra_father = await client.create_memory(
            topic="history_lyra_father_death",
            content=(
                "5 years ago: Lyra's father (guard captain before Marcus) died in bandit raid on eastern road. "
                "Marcus was his second-in-command. Marcus feels responsible - he should have scouted better. "
                "Thomas was there when Lyra got the news. Both men helped raise money for her. "
                "This is why Marcus is so protective and Thomas gave her a job. "
                "Lyra doesn't know Marcus blames himself."
            ),
            tags=["history", "lyra", "marcus", "thomas", "tragedy", "5years_ago"],
            confidence=0.96,
        )
        print("   ✅ Historical: Lyra's father's death connects Marcus-Lyra-Thomas")

        # Triangle: Elena-Silas-Marcus conflict
        _event_silas_arrived = await client.create_memory(
            topic="event_silas_arrival",
            content=(
                "6 months ago when Silas arrived: Marcus did background check - records seemed legitimate "
                "but something felt wrong. Elena welcomed competition initially. Silas was charming, "
                "offered to collaborate with Elena. She declined. He immediately turned aggressive. "
                "Marcus started surveillance after Elena reported Silas's sudden hostility. "
                "This created the current Marcus-investigates-Silas-who-targets-Elena dynamic."
            ),
            tags=["event", "silas", "elena", "marcus", "arrival", "6months_ago"],
            confidence=0.93,
        )
        print("   ✅ Event: Silas's arrival (affected Elena and Marcus)")

        # Multi-party: Weekly town gathering
        _routine_town_gathering = await client.create_memory(
            topic="routine_friday_gathering",
            content=(
                "Every Friday evening: Thomas hosts town gathering at the inn. Elena closes shop early to attend. "
                "Marcus stops by after guard shift. Lyra serves drinks and observes everyone. "
                "Silas started attending 4 months ago, sits in corner watching people. "
                "This is the social hub where information flows. Elena and Marcus always end up talking. "
                "Thomas watches them with knowing smiles. Lyra overhears everything. "
                "Silas takes mental notes of who talks to whom."
            ),
            tags=["routine", "friday", "all_npcs", "social_hub", "weekly"],
            confidence=0.94,
        )
        print("   ✅ Routine: Friday town gatherings (all NPCs interact)")

        # Location-based memories
        _location_town_square = await client.create_memory(
            topic="location_town_square",
            content=(
                "Town square sits between Elena's store, guard post, and inn. Everyone passes through daily. "
                "There's a fountain where people chat. Elena and Marcus often 'accidentally' meet here. "
                "Thomas can see the square from inn windows. Lyra runs errands through it. "
                "Silas's store overlooks it - he watches traffic patterns. "
                "The square is where information spreads fastest."
            ),
            tags=["location", "town_square", "central", "all_npcs"],
            confidence=0.92,
        )
        print("   ✅ Location: Town square (nexus of all NPC activity)")

        # Shared secret: Elena's financial troubles
        _secret_elena_finances = await client.create_memory(
            topic="secret_elena_struggling",
            content=(
                "Secret: Elena's store is struggling financially due to Silas's price war. "
                "She's too proud to ask for help. Thomas knows (he extended credit at inn). "
                "Marcus knows (he noticed her buying less). Lyra knows (Elena confided once). "
                "Only Silas doesn't know how effective his strategy is. "
                "Thomas and Marcus have been quietly trying to help without Elena noticing."
            ),
            tags=["secret", "elena", "finances", "known_by_multiple"],
            confidence=0.91,
        )
        print("   ✅ Secret: Elena's financial troubles (known by Thomas, Marcus, Lyra)")

        # Lyra's observations network
        _lyra_intelligence_network = await client.create_memory(
            topic="lyra_observation_journal",
            content=(
                "Lyra keeps a secret journal of observations: "
                "- Marcus visits Elena's store every 3 days (not random) "
                "- Silas meets mysterious people every Tuesday night "
                "- Thomas always seats Marcus where he can see Elena "
                "- Elena touches her wedding ring when Marcus talks to her "
                "- Silas watches the guard patrol schedule closely "
                "Lyra hasn't told anyone about her journal - it's her way of understanding the world."
            ),
            tags=["secret", "lyra", "observations", "all_npcs", "journal"],
            confidence=0.89,
        )
        print("   ✅ Secret: Lyra's observation journal (tracks all NPCs)")

        # Marcus's internal conflict
        _marcus_internal_conflict = await client.create_memory(
            topic="marcus_duty_vs_love",
            content=(
                "Marcus's constant struggle: duty to town vs personal feelings. "
                "He wants to confess to Elena but fears it'll distract him from protecting the town. "
                "He investigates Silas obsessively partly because Silas threatens Elena. "
                "Thomas tells him to follow his heart. Lyra reminds him of his humanity. "
                "Marcus worries he's becoming too hard, too focused on duty. "
                "Elena is the only thing that makes him feel human."
            ),
            tags=["internal", "marcus", "conflict", "duty_vs_love"],
            confidence=0.94,
        )
        print("   ✅ Internal: Marcus's duty vs. love conflict")

        # Thomas's role as social glue
        _thomas_mediator_role = await client.create_memory(
            topic="thomas_social_mediator",
            content=(
                "Thomas's unofficial role: he maintains social harmony in Rivertown. "
                "He counsels Marcus about Elena. He gives Elena business advice. "
                "He protects Lyra. He even tries to find good in Silas (unsuccessfully). "
                "Thomas knows everyone's secrets but never betrays confidence. "
                "People trust him because he's genuinely kind and neutral. "
                "The town would fall apart without Thomas's emotional labor."
            ),
            tags=["role", "thomas", "mediator", "social_glue", "all_npcs"],
            confidence=0.96,
        )
        print("   ✅ Role: Thomas as social mediator (connects all NPCs)")

        # Silas's hidden agenda
        _silas_true_goal = await client.create_memory(
            topic="silas_hidden_agenda",
            content=(
                "Silas's real goal (unknown to others): He's not just a merchant. "
                "He works for a larger bandit network expanding into the region. "
                "His mission: drive out local merchants (Elena), assess guard strength (Marcus), "
                "identify social influencers (Thomas), and recruit informants (tried with Lyra). "
                "The store is a front. He's mapping the town for future takeover. "
                "He underestimated Marcus's instincts and Lyra's perceptiveness."
            ),
            tags=["secret", "silas", "agenda", "criminal", "threat"],
            confidence=0.97,
        )
        print("   ✅ Secret: Silas's true agenda (bandit network)")

        # Past connection: Elena's late husband and Marcus
        _history_elena_husband_marcus = await client.create_memory(
            topic="history_blacksmith_marcus",
            content=(
                "Elena's late husband (blacksmith) and Marcus were close friends. "
                "The blacksmith knew Marcus loved Elena. Before he died (illness, 5 years ago), "
                "he told Marcus: 'Take care of her, and when enough time passes, tell her how you feel.' "
                "Marcus promised. But Marcus has been paralyzed by grief, guilt, and fear. "
                "He keeps the deathbed promise secret. Elena doesn't know her husband approved. "
                "This is Marcus's deepest pain - honoring his friend while loving his widow."
            ),
            tags=["history", "elena", "marcus", "husband", "deathbed_promise", "5years_ago"],
            confidence=0.98,
        )
        print("   ✅ Historical: Elena's husband's deathbed message to Marcus")

        # Lyra's secret skill
        _lyra_hidden_talent = await client.create_memory(
            topic="lyra_combat_training",
            content=(
                "Secret: Lyra's father trained her in combat before he died. "
                "She's not just an observant barmaid - she's dangerous with a knife. "
                "She practices in secret, honoring her father's memory. "
                "Marcus doesn't know. Thomas suspects but hasn't asked. "
                "This will be crucial when confronting Silas and bandits. "
                "Lyra is far more capable than anyone realizes."
            ),
            tags=["secret", "lyra", "combat", "hidden_skill"],
            confidence=0.90,
        )
        print("   ✅ Secret: Lyra's hidden combat skills (from her father)")

        # Elena's perception of the network
        _elena_awareness = await client.create_memory(
            topic="elena_social_awareness",
            content=(
                "What Elena knows about others: "
                "- Suspects Marcus has feelings but isn't sure "
                "- Knows Thomas is her most loyal friend "
                "- Fears Silas but doesn't know extent of threat "
                "- Treats Lyra like daughter, unaware of Lyra's combat skills "
                "- Carries guilt about not loving Marcus the way he might love her "
                "Elena is more perceptive than people think but struggles with emotional decisions."
            ),
            tags=["awareness", "elena", "perception", "all_npcs"],
            confidence=0.93,
        )
        print("   ✅ Awareness: Elena's perception of the social network")

        # Group memory: The bandit raid 3 years ago
        _history_bandit_raid_3years = await client.create_memory(
            topic="history_major_bandit_raid",
            content=(
                "3 years ago: Major bandit raid on Rivertown. Marcus led defense. "
                "Elena's store was targeted - Marcus personally protected it (revealing his feelings). "
                "Thomas's inn became field hospital. Lyra was 19, helped tend wounded. "
                "Town won but 12 people died. This is why everyone takes bandits seriously. "
                "It's why Marcus is paranoid about security. It's why the town trusts Marcus. "
                "Unknown to them: Silas's network is connected to those same bandits."
            ),
            tags=["history", "bandit_raid", "all_npcs", "trauma", "3years_ago"],
            confidence=0.96,
        )
        print("   ✅ Historical event: Major bandit raid 3 years ago (shaped everyone)")

        print(f"\n   📊 Total initial memories: 5 NPCs + 10 relationships + 15 historical/contextual")
        print(f"   📊 = 30 interconnected memory states!")
        print("   🕸️  Dense memory network with multiple layers of connections!")

        # ====================================================================
        # Day 1: Player Arrives + NPC Interactions
        # ====================================================================
        print("\n📅 Day 1: Player arrives & NPCs react...")
        print()

        # Player meets Elena
        elena_player_day1 = await client.create_memory(
            topic="elena_player_relationship",
            content=(
                "Player (adventurer named 'Aria') visited Elena's store for the first time. "
                "Elena was welcoming. Aria bought some basic supplies (rope, rations) "
                "and asked about the town. Elena shared information about local landmarks "
                "and warned about bandits on the eastern road. Aria was polite and paid fairly. "
                "Elena mentioned Aria to Thomas later - 'seems like a good person'."
            ),
            tags=["interaction", "elena", "player", "day1", "positive"],
            confidence=0.9,
            custom_metadata={"day": 1, "player_action": "friendly_purchase"},
        )
        print("   💬 Player meets Elena (positive)")

        # Player meets Marcus
        marcus_player_day1 = await client.create_memory(
            topic="marcus_player_relationship",
            content=(
                "Player 'Aria' approached Marcus at the guard post. Marcus was suspicious "
                "of the new adventurer and questioned their purpose in town. Aria explained "
                "they're just passing through and looking for work. Marcus gave a stern warning "
                "about following town laws. Aria agreed respectfully. Marcus told Elena later "
                "to 'be careful with this stranger'."
            ),
            tags=["interaction", "marcus", "player", "day1", "neutral"],
            confidence=0.85,
            custom_metadata={"day": 1, "player_action": "respectful_compliance"},
        )
        print("   💬 Player meets Marcus (suspicious)")

        # NPCs talk ABOUT the player
        _thomas_hears_about_player = await client.create_memory(
            topic="thomas_town_knowledge",
            content=(
                "Thomas heard from both Elena and Marcus about the new adventurer 'Aria'. "
                "Elena speaks well of them, Marcus is cautious. Thomas's opinion: "
                "'Let's give them a chance'. He prepared a room at the inn in case Aria needs lodging."
            ),
            tags=["npc_communication", "thomas", "day1"],
            confidence=0.88,
        )
        print("   🗨️  Thomas hears about player from Elena & Marcus\n")

        # ====================================================================
        # Day 2: NPC-to-NPC Conflict (Silas vs Elena)
        # ====================================================================
        print("📅 Day 2: Merchant rivalry escalates...")
        print()

        silas_undercuts_elena = await client.create_memory(
            topic="npc_silas_merchant",
            content=(
                "Silas started a price war with Elena. He's selling basic goods at 30% below "
                "her prices - clearly trying to drive her out of business. He's spreading "
                "rumors that Elena's goods are lower quality. This is hurting Elena financially. "
                "Marcus noticed and is investigating whether Silas is doing something illegal."
            ),
            tags=["npc", "merchant", "silas", "rival", "conflict", "day2"],
            confidence=0.92,
        )
        print("   ⚔️  Silas escalates price war against Elena")
        print("   🔄 Waiting for cascading evolution...\n")
        await asyncio.sleep(3)  # Give cascading evolution time to work

        # ====================================================================
        # Day 3: Major Event - Lyra discovers Silas's secret!
        # ====================================================================
        print("📅 Day 3: MAJOR DISCOVERY - This should trigger cascading evolution!")
        print()

        lyra_discovers_secret = await client.create_memory(
            topic="npc_lyra_barmaid",
            content=(
                "CRITICAL DISCOVERY: Lyra overheard Silas meeting with bandits in a back room "
                "of the inn late at night. Silas is the one SUPPLYING the bandits! "
                "He's using them to attack trade caravans, then buying stolen goods cheap. "
                "That's how he undercuts Elena's prices. Lyra is scared but tells Thomas. "
                "This revelation will change everything in Rivertown."
            ),
            tags=["npc", "barmaid", "lyra", "discovery", "critical", "day3"],
            confidence=0.98,
        )
        print("   🚨 Lyra discovers Silas is working with bandits!")
        print("   ⚠️  This should CASCADE to: Thomas, Marcus, Elena, Silas relationships")
        print("   🔄 Waiting for cascading evolution to propagate...\n")
        await asyncio.sleep(4)  # More time for complex cascading

        # ====================================================================
        # Phase 3: Check Cascading Evolution
        # ====================================================================
        print("🔍 Phase 3: Verifying cascading evolution...")
        print()

        topics_to_check = [
            "npc_thomas_innkeeper",
            "npc_marcus_guard",
            "npc_elena_merchant",
            "relationship_elena_thomas",
            "npc_silas_merchant",
        ]

        evolved_count = 0
        for topic in topics_to_check:
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

        print(f"\n   📊 Evolution summary: {evolved_count}/{len(topics_to_check)} topics evolved\n")

        # ====================================================================
        # Day 4: Thomas tells Marcus + Marcus investigates
        # ====================================================================
        print("📅 Day 4: Information spreads...")
        print()

        thomas_tells_marcus = await client.create_memory(
            topic="thomas_marcus_communication",
            content=(
                "Thomas immediately told Marcus about what Lyra discovered. "
                "Marcus was furious - he KNEW Silas was suspicious! "
                "Marcus is organizing a raid on Silas's warehouse tonight. "
                "He asked Thomas to keep Lyra safe and not let Silas know they're onto him."
            ),
            tags=["npc_communication", "thomas", "marcus", "day4", "investigation"],
            confidence=0.95,
        )
        print("   🗣️  Thomas informs Marcus about Lyra's discovery")

        # Marcus talks to Elena
        marcus_warns_elena = await client.create_memory(
            topic="marcus_elena_relationship",
            content=(
                "Marcus visited Elena's shop to warn her about Silas. "
                "He told her that Silas is working with bandits and will be arrested tonight. "
                "Elena was shocked and grateful. Marcus stayed to make sure she was safe. "
                "Elena noticed Marcus was being especially protective and caring. "
                "For the first time, she sensed his deeper feelings for her."
            ),
            tags=["relationship", "marcus", "elena", "day4", "romantic_tension"],
            confidence=0.90,
        )
        print("   💝 Marcus protects Elena - romantic tension building")

        # ====================================================================
        # Day 5: Player helps in raid (player-NPC-NPC triangle)
        # ====================================================================
        print("\n📅 Day 5: Raid on Silas's warehouse...")
        print()

        # Player helps Marcus
        player_helps_raid = await client.create_memory(
            topic="marcus_player_relationship",
            content=(
                "Aria volunteered to help Marcus with the raid on Silas's warehouse. "
                "During the raid, bandits ambushed them. Aria saved Marcus's life by "
                "blocking a sword strike. Marcus was deeply grateful. After the raid succeeded "
                "and Silas was arrested, Marcus told Aria: 'You're not a stranger anymore - "
                "you're one of us.' He introduced Aria to Elena as a hero."
            ),
            tags=["interaction", "marcus", "player", "day5", "combat", "trust"],
            confidence=0.96,
            custom_metadata={"day": 5, "player_action": "heroic_defense"},
        )
        print("   ⚔️  Player saves Marcus's life in raid")

        # Elena's reaction to player helping
        elena_player_day5 = await client.create_memory(
            topic="elena_player_relationship",
            content=(
                "Marcus brought Aria to Elena's shop after the raid and explained how "
                "Aria saved his life. Elena was overwhelmed with gratitude - Marcus is "
                "very important to her. She gave Aria a family heirloom (a silver pendant) "
                "as thanks. She said: 'You saved someone precious to me.' "
                "Aria is now like family to Elena."
            ),
            tags=["interaction", "elena", "player", "day5", "gratitude", "family"],
            confidence=0.95,
            custom_metadata={"day": 5, "relationship_level": "family_like"},
        )
        print("   💎 Elena gives player family heirloom")

        # Thomas celebrates at inn
        _celebration = await client.create_memory(
            topic="thomas_town_knowledge",
            content=(
                "Thomas threw a celebration at the inn after Silas was arrested. "
                "The whole town came. Thomas toasted Aria, Marcus, and Lyra as heroes. "
                "He whispered to Marcus: 'You should tell Elena how you feel - life's too short.' "
                "The town's mood has completely changed - people feel safe again."
            ),
            tags=["event", "thomas", "celebration", "day5"],
            confidence=0.93,
        )
        print("   🍺 Thomas hosts celebration at inn\n")

        # ====================================================================
        # Day 7: Marcus confesses to Elena (major relationship shift)
        # ====================================================================
        print("📅 Day 7: Major relationship development...")
        print()

        marcus_confesses = await client.create_memory(
            topic="marcus_elena_relationship",
            content=(
                "Marcus finally confessed his feelings to Elena. He told her he's been "
                "in love with her for years but didn't want to rush her after her husband's death. "
                "Elena was surprised but admitted she had growing feelings for him too. "
                "They agreed to take things slow. Thomas is thrilled for them. "
                "Lyra thinks it's the most romantic thing ever. "
                "The whole town is gossiping about it (in a good way)."
            ),
            tags=["relationship", "marcus", "elena", "day7", "romance", "confession"],
            confidence=0.97,
        )
        print("   💕 Marcus confesses - Elena accepts!")
        print("   🔄 This should trigger more cascading evolution...\n")
        await asyncio.sleep(3)

        # ====================================================================
        # Phase 5: Manual Promotion - Synthesize Town Social Dynamics
        # ====================================================================
        print("🚀 Phase 5: Manual promotion - synthesizing town dynamics...")
        print()

        promotion = await client.promote_memory(
            topic="npc_elena_merchant",
            reason=(
                "Week 1 complete: Major events have transformed Rivertown's social dynamics. "
                "Elena went from isolated merchant to central figure in love story and "
                "crime resolution. Need comprehensive synthesis of her evolved state and relationships."
            ),
        )

        if promotion.success:
            new_state = promotion.new_state
            print(f"   ✅ Promotion successful!")
            print(f"   📊 New version: v{new_state.version}")
            print(f"   📝 Synthesized content:")
            print(f"      {new_state.content[:400]}...")
            print(f"\n   📚 Sources: {len(promotion.synthesis_sources)}")
            print(f"   🎯 Confidence: {promotion.confidence:.2f}")
            if promotion.reasoning:
                print(f"   💭 Reasoning: {promotion.reasoning[:200]}...")
        else:
            print(f"   ❌ Failed: {promotion.error_message}")

        # ====================================================================
        # Phase 6: Complex Queries - Testing Relationship Understanding
        # ====================================================================
        print("\n🔎 Phase 6: Complex relationship queries...")
        print()

        queries = [
            "How did the town discover Silas was working with bandits?",
            "What is the relationship network between Elena, Marcus, Thomas, and Lyra?",
            "How did the player earn the trust of the townspeople?",
            "What role did Lyra play in resolving the crisis?",
        ]

        for i, query_text in enumerate(queries, 1):
            print(f"\n   Query {i}: '{query_text}'")
            result = await client.query(
                query_text,
                include_history=True,
                include_related=True,
            )
            print(f"   Answer: {result.answer[:250]}...")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Sources: {len(result.sources)}")

        # ====================================================================
        # Phase 7: View Evolution Traces
        # ====================================================================
        print("\n📈 Phase 7: Viewing NPC evolution traces...")
        print()

        key_npcs = [
            "npc_elena_merchant",
            "npc_marcus_guard",
            "npc_lyra_barmaid",
            "marcus_elena_relationship",
        ]

        for topic in key_npcs:
            trace = await client.get_trace(topic)
            print(f"\n   📌 {topic}: {len(trace)} version(s)")
            for state in trace[:3]:  # Show first 3
                tags_str = ", ".join(state.tags[:4])
                cascading = "🔄" if "cascading_evolved" in state.tags else ""
                print(f"      v{state.version} {cascading} - {state.timestamp.strftime('%H:%M:%S')}")
                print(f"         Tags: [{tags_str}]")
                print(f"         {state.content[:100]}...")

        print("\n" + "=" * 70)
        print("✅ Complex NPC simulation completed!")
        print("=" * 70)
        print("\nKey Verifications:")
        print("  1. ✓ Multiple NPCs with interconnected relationships")
        print("  2. ✓ NPC-to-NPC interactions (not just NPC-to-player)")
        print("  3. ✓ Cascading evolution when one NPC's state affects others")
        print("  4. ✓ Complex social dynamics and information propagation")
        print("  5. ✓ Manual promotion synthesizes comprehensive character state")
        print("  6. ✓ Queries understand multi-character relationships")
        print()


if __name__ == "__main__":
    try:
        asyncio.run(simulate_npc_interactions())
    except KeyboardInterrupt:
        print("\n\n⚠️  Example cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("   1. TracingRAG API is running")
        print("   2. All services are healthy: poetry run python scripts/verify_setup.py")
        sys.exit(1)
