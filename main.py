#!/usr/bin/env python3
"""
Quick test run with CSV mockup data
"""

import sys
import pandas as pd
from datetime import datetime

sys.path.insert(0, 'scripts')
from scripts.optimizer_waterflow import HydrexVotingOptimizer

def main():
    print("\n" + "="*80)
    print(" 🧪 HYDREX OPTIMIZER - TEST RUN WITH MOCKUP DATA")
    print("="*80 + "\n")
    
    # Load test data - SIMPLIFIED (current epoch only)
    print("📂 Loading test data from CSV...")
    df_pools = pd.read_csv('data/weekly_pools.csv')


    print(f"✅ Loaded {len(df_pools)} pools")
    print(f"   Total votes: {pd.to_numeric(df_pools['current_votes'], errors='coerce').sum()/1e6:.1f}M")
    print(f"   Total epoch rewards: ${pd.to_numeric(df_pools['projected_rewards'], errors='coerce').sum():,.0f}")
    print(f"   Average pool TVL: ${pd.to_numeric(df_pools['tvl_usd'], errors='coerce').mean()/1e6:.2f}M")
    
    # Configuration - FIXED
    USER_VOTING_POWER = 1_000_000  # 1M votes
    
    total_votes_numeric = float(pd.to_numeric(df_pools['current_votes'], errors='coerce').sum())
    config = {
        'max_pools': 20,
        'max_pool_allocation_pct': 0.40,
        'min_pool_allocation': 10000,
        'saturation_threshold': 0.25,
        'total_system_votes': 168000000,  # ✅ FIXED
        'min_tvl_threshold': 100000,
        'max_partner_pools': 2
    }
    total_system_votes = config['total_system_votes']
    
    print("⚙️  Configuration:")
    print(f"   User voting power: {USER_VOTING_POWER:,} ({USER_VOTING_POWER/total_system_votes*100:.2f}% of total)")
    print(f"   Max pools to select: {config['max_pools']}")
    print(f"   Max per pool: {config['max_pool_allocation_pct']*100:.0f}% = {USER_VOTING_POWER*config['max_pool_allocation_pct']:,.0f} votes")
    print(f"   Min per pool: {config['min_pool_allocation']:,} votes\n")
    
    # Show initial pool analysis - SHOWS POOL NAMES ✅
    print("="*80)
    print(" 📊 INITIAL POOL ANALYSIS")
    print("="*80)
    print(f"{'Pool':<15} {'TVL':>9} {'Rewards':>10} {'Votes':>12} {'Vote %':>9} {'Rewards/Vote':>14}")
    print("-"*85)

    for _, pool in df_pools.iterrows():
        # ✅ SHOW POOL NAME FIRST (same logic as optimizer)
        pool_name = pool.get('pool_name', str(pool['pool_id'])[:12])
    
        # ✅ FIXED: Safe numeric conversion everywhere
        current_votes_num = pd.to_numeric(pool['current_votes'], errors='coerce')
        projected_rewards_num = pd.to_numeric(pool['projected_rewards'], errors='coerce')
        tvl_num = pd.to_numeric(pool['tvl_usd'], errors='coerce')
    
        vote_pct = current_votes_num / total_votes_numeric * 100
        rewards_per_vote = projected_rewards_num / current_votes_num * 1000 if current_votes_num > 0 else 0
    
        print(f"{pool_name:<15} {tvl_num/1e6:>8.1f}M "
              f"{projected_rewards_num:>9,.0f} $ {current_votes_num/1e6:>10.1f}M "
              f"{vote_pct:>7.1f}% {rewards_per_vote:>11.4f}$")

    print("="*85 + "\n")
    
    # Run optimizer
    print("🎯 Running optimization algorithm...")
    print("   (This finds pools with best marginal returns for your votes)\n")
    
    optimizer = HydrexVotingOptimizer(config)
    allocation, results_df = optimizer.optimize_allocation(df_pools, USER_VOTING_POWER)
    
    # Display results - FIXED ✅
    print("="*80)
    print(" ✨ OPTIMAL ALLOCATION RESULTS")
    print("="*80)
    print(f"{'Pool':<15} {'Your Votes':>12} {'% Alloc':>8} {'Your Share':>10} {'Proj. Fees':>12} {'Marg. APR':>10}")
    print("-"*80)
    
    total_projected_fees = 0
    
    for _, row in results_df.iterrows():
        pool_match = df_pools[df_pools['pool_id'] == row['pool_id']]
        if not pool_match.empty:
            pool = pool_match.iloc[0]
            # ✅ FIXED: Safe numeric conversion
            current_votes_num = pd.to_numeric(pool['current_votes'], errors='coerce')
            new_total_votes = current_votes_num + row['votes_allocated']
            your_share = row['votes_allocated'] / new_total_votes * 100 if new_total_votes > 0 else 0
            
            pool_name = row.get('pair', row['pool_id'][:12])  # Use optimizer's pool name
            print(f"{pool_name:<15} {row['votes_allocated']:>12,.0f} "
                  f"{row['allocation_pct']:>7.1f}% {your_share:>9.2f}% "
                  f"${row['projected_fees']:>11,.2f} {row['marginal_apr']:>9.1f}%")
            
            total_projected_fees += row['projected_fees']
    
    print("-"*80)
    print(f"{'TOTAL':<15} {USER_VOTING_POWER:>12,} {100.0:>7.1f}% {'':>10} ${total_projected_fees:>11,.2f}")
    print("="*80 + "\n")
    
    # Show what we're NOT voting for and why - FIXED ✅
    unselected = df_pools[~df_pools['pool_id'].isin(allocation.keys())]
    
    if len(unselected) > 0:
        print("="*80)
        print(" ❌ POOLS NOT SELECTED (Why?)")
        print("="*80)
        
        for _, pool in unselected.iterrows():
            # ✅ FIXED: Safe numeric conversion
            current_votes_num = pd.to_numeric(pool['current_votes'], errors='coerce')
            projected_rewards_num = pd.to_numeric(pool['projected_rewards'], errors='coerce')
            vote_share = current_votes_num / total_votes_numeric
            rewards_per_1000 = projected_rewards_num / current_votes_num * 1000 if current_votes_num > 0 else 0
            
            reasons = []
            if vote_share > config['saturation_threshold']:
                reasons.append("Over-saturated (too many votes)")
            if rewards_per_1000 < 0.5:
                reasons.append("Poor rewards/vote ratio")
            if pd.to_numeric(pool['tvl_usd'], errors='coerce') < 1e6:
                reasons.append("Low TVL (risky)")
            if not reasons:
                reasons.append("Lower marginal returns than selected pools")
            
            print(f"• {pool['pool_name']:<15} - {', '.join(reasons)}")
        
        print("="*80 + "\n")
    
    # Run Monte Carlo simulation
    print("📈 Running Monte Carlo simulation (1000 iterations)...")
    sim_results = optimizer.simulate_returns(df_pools, allocation, n_simulations=1000)
    
    print("\n" + "="*80)
    print(" 🎲 EXPECTED RETURNS (Monte Carlo Simulation)")
    print("="*80)
    print(f"Expected Weekly Return:       ${sim_results['mean']:>10,.2f}")
    print(f"Median:                       ${sim_results['median']:>10,.2f}")
    print(f"Standard Deviation:           ${sim_results['std']:>10,.2f}")
    print(f"Best Case (95th percentile):  ${sim_results['p95']:>10,.2f}")
    print(f"Worst Case (5th percentile):  ${sim_results['p5']:>10,.2f}")
    print(f"Sharpe Ratio:                 {sim_results['sharpe']:>11.2f}")
    print("-"*80)
    
    weekly_apr = (sim_results['mean'] / USER_VOTING_POWER) * 100
    annual_apr = weekly_apr * 52
    
    print(f"Weekly APR:                   {weekly_apr:>10.2f}%")
    print(f"Annualized APR:               {annual_apr:>10.1f}%")
    print("="*80 + "\n")
    
    # Comparison with naive strategy - FIXED ✅
    print("="*80)
    print(" 📊 STRATEGY COMPARISON")
    print("="*80)
    
    # Naive strategy: put all votes in highest APR pool
    df_pools['current_apr'] = (
        pd.to_numeric(df_pools['projected_rewards'], errors='coerce') / 
        pd.to_numeric(df_pools['current_votes'], errors='coerce')
    ) * 52 * 100
    best_apr_pool = df_pools.loc[df_pools['current_apr'].idxmax()]
    
    naive_total_votes = pd.to_numeric(best_apr_pool['current_votes'], errors='coerce') + USER_VOTING_POWER
    naive_share = USER_VOTING_POWER / naive_total_votes
    naive_return = pd.to_numeric(best_apr_pool['projected_rewards'], errors='coerce') * naive_share
    
    print(f"NAIVE STRATEGY (all votes → {best_apr_pool['pool_id']}):")
    print(f"   Expected weekly return: ${naive_return:,.2f}")
    print(f"   Annual APR: {(naive_return / USER_VOTING_POWER * 52 * 100):.1f}%\n")
    
    print(f"OPTIMIZED STRATEGY (diversified across {len(allocation)} pools):")
    print(f"   Expected weekly return: ${sim_results['mean']:,.2f}")
    print(f"   Annual APR: {annual_apr:.1f}%\n")
    
    improvement = ((sim_results['mean'] - naive_return) / naive_return) * 100
    print(f"💰 IMPROVEMENT: +{improvement:.1f}% better returns!")
    print(f"   Risk-adjusted (Sharpe): {sim_results['sharpe']:.2f} vs ~{naive_return/(naive_return*0.15):.2f}")
    print("="*80 + "\n")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f'data/Epoch_allocation_{timestamp}.csv', index=False)
    
    print(f"💾 Results saved to: data/test_allocation_{timestamp}.csv")
    
    # Generate vote transaction
    print("\n" + "="*80)
    print(" 🗳️  VOTE TRANSACTION DATA")
    print("="*80)
    print("Copy this data to submit your votes:\n")
    
    for pool_id, votes in allocation.items():
        print(f"   {pool_id}: {int(votes):,} votes")
    
    print("\n" + "="*80)
    print(" ✅ TEST COMPLETE!")
    print("="*80)
    print("\nKey Takeaways:")
    print(f"1. Algorithm selected {len(allocation)} pools for optimal diversification")
    print(f"2. Expected to earn ${sim_results['mean']:,.2f}/week (${sim_results['mean']*52:,.0f}/year)")
    print(f"3. {improvement:.1f}% better than naive 'highest APR' strategy")
    print("\n")
    
    return allocation, results_df, sim_results


if __name__ == "__main__":
    try:
        allocation, results, simulation = main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
