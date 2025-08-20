import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

def display_advanced_metrics(analysis_results):
    """Display comprehensive metrics dashboard for model comparison results."""
    
    st.header("Advanced Evaluation Metrics")
    
    # Extract metrics data
    metrics_data = []
    for model, result in analysis_results.results.items():
        # Get additional metrics if available
        if hasattr(result, 'metrics'):
            row = {
                'Model': model,
                'Overall Score': result.overall_score,
                'Token Count': result.metrics.get('token_count', 0),
                'Character Count': result.metrics.get('character_count', 0),
                'Readability Score': result.metrics.get('readability_score', 0),
                'Complexity Score': result.metrics.get('complexity_score', 0),
                'Coverage Score': result.metrics.get('coverage_score', 0),
                'Jargon Penalty': result.metrics.get('jargon_penalty', 0)
            }
            # Add individual criterion scores
            row.update(result.scores)
            metrics_data.append(row)
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Display metrics in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Core Metrics", "Quality Analysis", "Readability", "Advanced Metrics"])
    
    with tab1:
        st.subheader("Core Performance Metrics")
        
        # Main performance indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            best_overall = metrics_df.loc[metrics_df['Overall Score'].idxmax()]
            st.metric(
                "Best Overall Score",
                f"{best_overall['Overall Score']:.2f}/10",
                delta=f"+{best_overall['Overall Score'] - metrics_df['Overall Score'].min():.2f}"
            )
        
        with col2:
            avg_score = metrics_df['Overall Score'].mean()
            st.metric(
                "Average Score",
                f"{avg_score:.2f}/10",
                delta=f"±{metrics_df['Overall Score'].std():.2f}"
            )
        
        with col3:
            best_accuracy = metrics_df.loc[metrics_df['accuracy'].idxmax()]
            st.metric(
                "Best Accuracy",
                f"{best_accuracy['accuracy']:.1f}/10",
                delta=f"Model: {best_accuracy['Model']}"
            )
        
        with col4:
            best_helpfulness = metrics_df.loc[metrics_df['helpfulness'].idxmax()]
            st.metric(
                "Most Helpful",
                f"{best_helpfulness['helpfulness']:.1f}/10",
                delta=f"Model: {best_helpfulness['Model']}"
            )
        
        # Detailed scores table
        st.subheader("Detailed Scoring Breakdown")
        
        # Format the dataframe for better display
        display_df = metrics_df[['Model', 'Overall Score', 'accuracy', 'completeness', 
                                'helpfulness', 'clarity', 'relevance', 'professionalism']].copy()
        
        # Style the dataframe
        styled_df = display_df.style.format({
            'Overall Score': '{:.2f}',
            'accuracy': '{:.1f}',
            'completeness': '{:.1f}',
            'helpfulness': '{:.1f}',
            'clarity': '{:.1f}',
            'relevance': '{:.1f}',
            'professionalism': '{:.1f}'
        }).background_gradient(subset=['Overall Score'], cmap='RdYlGn', vmin=0, vmax=10)
        
        st.dataframe(styled_df, use_container_width=True)
    
    with tab2:
        st.subheader("Quality Analysis")
        
        # Radar chart for quality dimensions
        criteria = ['accuracy', 'completeness', 'helpfulness', 'clarity', 'relevance', 'professionalism']
        
        fig_radar = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, (_, row) in enumerate(metrics_df.iterrows()):
            values = [row[criterion] for criterion in criteria]
            values.append(values[0])  # Close the polygon
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=criteria + [criteria[0]],
                fill='toself',
                name=row['Model'],
                line_color=colors[i % len(colors)],
                opacity=0.7
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    tickmode='linear',
                    tick0=0,
                    dtick=2
                )
            ),
            title="Quality Dimensions Comparison",
            showlegend=True,
            height=500
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Quality insights
        st.subheader("Quality Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Strengths analysis
            st.write("**Model Strengths:**")
            for _, row in metrics_df.iterrows():
                strengths = []
                for criterion in criteria:
                    if row[criterion] >= 8.0:
                        strengths.append(criterion.title())
                
                if strengths:
                    st.write(f"• **{row['Model']}**: {', '.join(strengths)}")
                else:
                    st.write(f"• **{row['Model']}**: Consistent performance across criteria")
        
        with col2:
            # Improvement areas
            st.write("**Areas for Improvement:**")
            for _, row in metrics_df.iterrows():
                weaknesses = []
                for criterion in criteria:
                    if row[criterion] < 6.0:
                        weaknesses.append(criterion.title())
                
                if weaknesses:
                    st.write(f"• **{row['Model']}**: {', '.join(weaknesses)}")
                else:
                    st.write(f"• **{row['Model']}**: No significant weaknesses identified")
    
    with tab3:
        st.subheader("Readability & Communication Analysis")
        
        # Response length comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig_length = px.bar(
                metrics_df, 
                x='Model', 
                y='Token Count',
                title="Response Length Comparison",
                color='Token Count',
                color_continuous_scale='viridis'
            )
            fig_length.update_layout(height=400)
            st.plotly_chart(fig_length, use_container_width=True)
        
        with col2:
            fig_readability = px.bar(
                metrics_df,
                x='Model',
                y='Readability Score',
                title="Readability Score (Higher = Easier to Read)",
                color='Readability Score',
                color_continuous_scale='RdYlGn'
            )
            fig_readability.update_layout(height=400)
            st.plotly_chart(fig_readability, use_container_width=True)
        
        # Communication quality metrics
        st.subheader("Communication Quality Breakdown")
        
        communication_metrics = ['clarity', 'professionalism', 'Readability Score', 'Jargon Penalty']
        available_metrics = [m for m in communication_metrics if m in metrics_df.columns]
        
        if available_metrics:
            comm_df = metrics_df[['Model'] + available_metrics].copy()
            
            # Create subplot for communication metrics
            fig_comm = make_subplots(
                rows=2, cols=2,
                subplot_titles=available_metrics,
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            for i, metric in enumerate(available_metrics[:4]):
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                fig_comm.add_trace(
                    go.Bar(
                        x=comm_df['Model'],
                        y=comm_df[metric],
                        name=metric,
                        showlegend=False,
                        marker_color=px.colors.qualitative.Set3[i]
                    ),
                    row=row, col=col
                )
            
            fig_comm.update_layout(height=600, title_text="Communication Quality Metrics")
            st.plotly_chart(fig_comm, use_container_width=True)
    
    with tab4:
        st.subheader("Advanced Performance Metrics")
        
        # Complexity vs Coverage analysis
        if 'Complexity Score' in metrics_df.columns and 'Coverage Score' in metrics_df.columns:
            fig_scatter = px.scatter(
                metrics_df,
                x='Coverage Score',
                y='Complexity Score',
                size='Overall Score',
                color='Model',
                title="Complexity vs Coverage Analysis",
                labels={
                    'Coverage Score': 'Test Case Coverage (0-1)',
                    'Complexity Score': 'Complexity Handling (0-1)'
                },
                hover_data=['Overall Score']
            )
            
            # Add diagonal reference line
            fig_scatter.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    name='Ideal Balance',
                    showlegend=True
                )
            )
            
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Performance efficiency metrics
        st.subheader("Efficiency Metrics")
        
        efficiency_cols = st.columns(3)
        
        with efficiency_cols[0]:
            if 'Token Count' in metrics_df.columns:
                avg_tokens = metrics_df['Token Count'].mean()
                most_efficient = metrics_df.loc[metrics_df['Token Count'].idxmin()]
                st.metric(
                    "Most Efficient (Tokens)",
                    f"{most_efficient['Token Count']} tokens",
                    delta=f"{most_efficient['Model']}"
                )
        
        with efficiency_cols[1]:
            # Score per token efficiency
            if 'Token Count' in metrics_df.columns:
                metrics_df['Score_per_Token'] = metrics_df['Overall Score'] / metrics_df['Token Count']
                best_efficiency = metrics_df.loc[metrics_df['Score_per_Token'].idxmax()]
                st.metric(
                    "Best Score/Token Ratio",
                    f"{best_efficiency['Score_per_Token']:.4f}",
                    delta=f"{best_efficiency['Model']}"
                )
        
        with efficiency_cols[2]:
            if 'Jargon Penalty' in metrics_df.columns:
                lowest_jargon = metrics_df.loc[metrics_df['Jargon Penalty'].idxmin()]
                st.metric(
                    "Lowest Jargon Penalty",
                    f"{lowest_jargon['Jargon Penalty']:.3f}",
                    delta=f"{lowest_jargon['Model']}"
                )
        
        # Advanced metrics correlation heatmap
        st.subheader("Metrics Correlation Analysis")
        
        numeric_columns = metrics_df.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = metrics_df[numeric_columns].corr()
        
        fig_heatmap = px.imshow(
            correlation_matrix,
            title="Metrics Correlation Heatmap",
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Interpretation of correlations
        st.write("**Correlation Insights:**")
        strong_correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    metric1 = correlation_matrix.columns[i]
                    metric2 = correlation_matrix.columns[j]
                    strong_correlations.append({
                        'metric1': metric1,
                        'metric2': metric2,
                        'correlation': corr_value
                    })
        
        if strong_correlations:
            for corr in strong_correlations:
                direction = "positively" if corr['correlation'] > 0 else "negatively"
                st.write(f"• {corr['metric1']} is strongly {direction} correlated with {corr['metric2']} (r={corr['correlation']:.2f})")
        else:
            st.write("• No strong correlations detected between metrics")

def display_response_analysis(analysis_results):
    """Display detailed analysis of model responses."""
    
    st.header("Response Content Analysis")
    
    # Create tabs for each model
    model_tabs = st.tabs([model for model in analysis_results.results.keys()])
    
    for i, (model, result) in enumerate(analysis_results.results.items()):
        with model_tabs[i]:
            st.subheader(f"{model} Analysis")
            
            # Response preview
            with st.expander("Full Response", expanded=False):
                st.write(result.response)
            
            # Key metrics for this response
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Overall Score", f"{result.overall_score:.1f}/10")
            
            with col2:
                best_criterion = max(result.scores.items(), key=lambda x: x[1])
                st.metric("Strongest Area", f"{best_criterion[0].title()}", f"{best_criterion[1]:.1f}/10")
            
            with col3:
                worst_criterion = min(result.scores.items(), key=lambda x: x[1])
                st.metric("Improvement Area", f"{worst_criterion[0].title()}", f"{worst_criterion[1]:.1f}/10")
            
            # Response characteristics
            st.subheader("Response Characteristics")
            
            if hasattr(result, 'metrics'):
                char_col1, char_col2 = st.columns(2)
                
                with char_col1:
                    st.write("**Length Metrics:**")
                    st.write(f"• Word count: {result.metrics.get('token_count', 'N/A')}")
                    st.write(f"• Character count: {result.metrics.get('character_count', 'N/A')}")
                    st.write(f"• Readability score: {result.metrics.get('readability_score', 'N/A'):.1f}")
                
                with char_col2:
                    st.write("**Quality Metrics:**")
                    st.write(f"• Complexity handling: {result.metrics.get('complexity_score', 'N/A'):.2f}")
                    st.write(f"• Coverage score: {result.metrics.get('coverage_score', 'N/A'):.2f}")
                    st.write(f"• Jargon penalty: {result.metrics.get('jargon_penalty', 'N/A'):.3f}")
            
            # Score breakdown chart
            scores_df = pd.DataFrame([
                {'Criterion': k.title(), 'Score': v} 
                for k, v in result.scores.items()
            ])
            
            fig_bar = px.bar(
                scores_df,
                x='Criterion',
                y='Score',
                title=f"{model} - Detailed Score Breakdown",
                color='Score',
                color_continuous_scale='RdYlGn',
                range_color=[0, 10]
            )
            
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

def display_comparative_insights(analysis_results):
    """Display comparative insights and recommendations."""
    
    st.header("Comparative Analysis & Insights")
    
    models = list(analysis_results.results.keys())
    
    if len(models) == 2:
        model1, model2 = models
        result1 = analysis_results.results[model1]
        result2 = analysis_results.results[model2]
        
        st.subheader(f"{model1} vs {model2}")
        
        # Head-to-head comparison
        comparison_data = []
        for criterion in result1.scores.keys():
            score1 = result1.scores[criterion]
            score2 = result2.scores[criterion]
            winner = model1 if score1 > score2 else model2
            margin = abs(score1 - score2)
            
            comparison_data.append({
                'Criterion': criterion.title(),
                f'{model1}': score1,
                f'{model2}': score2,
                'Winner': winner,
                'Margin': margin
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Style the comparison table
        def highlight_winner(row):
            model1_col = f'{model1}'
            model2_col = f'{model2}'
            
            if row[model1_col] > row[model2_col]:
                return [f'background-color: lightgreen' if col == model1_col 
                       else f'background-color: lightcoral' if col == model2_col 
                       else '' for col in row.index]
            else:
                return [f'background-color: lightcoral' if col == model1_col 
                       else f'background-color: lightgreen' if col == model2_col 
                       else '' for col in row.index]
        
        styled_comparison = comparison_df.style.apply(highlight_winner, axis=1)
        st.dataframe(styled_comparison, use_container_width=True)
        
        # Victory summary
        model1_wins = sum(1 for row in comparison_data if row['Winner'] == model1)
        model2_wins = sum(1 for row in comparison_data if row['Winner'] == model2)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"{model1} Wins", model1_wins)
        
        with col2:
            st.metric(f"{model2} Wins", model2_wins)
        
        with col3:
            avg_margin = sum(row['Margin'] for row in comparison_data) / len(comparison_data)
            st.metric("Avg Margin", f"{avg_margin:.2f}")
    
    # Recommendations
    st.subheader("Recommendations")
    
    winner = analysis_results.winner
    winner_result = analysis_results.results[winner]
    
    st.success(f"**Recommended Model: {winner}** (Score: {winner_result.overall_score:.1f}/10)")
    
    # Detailed recommendations
    recommendations = []
    
    # Based on overall performance
    if winner_result.overall_score >= 8.0:
        recommendations.append(f"{winner} demonstrates excellent performance across all criteria and is suitable for production use.")
    elif winner_result.overall_score >= 7.0:
        recommendations.append(f"{winner} shows good performance but may benefit from prompt optimization in weaker areas.")
    else:
        recommendations.append(f"{winner} performs adequately but consider testing additional models or refining prompts.")
    
    # Based on specific strengths
    best_criterion = max(winner_result.scores.items(), key=lambda x: x[1])
    if best_criterion[1] >= 9.0:
        recommendations.append(f"Leverage {winner}'s exceptional {best_criterion[0]} capabilities for tasks requiring this skill.")
    
    # Based on weaknesses
    worst_criterion = min(winner_result.scores.items(), key=lambda x: x[1])
    if worst_criterion[1] < 6.0:
        recommendations.append(f"Address {winner}'s {worst_criterion[0]} weaknesses through targeted prompt engineering or additional training data.")
    
    for rec in recommendations:
        st.write(f"• {rec}")
    
    # Use case recommendations
    st.subheader("Use Case Recommendations")
    
    for model, result in analysis_results.results.items():
        st.write(f"**{model}:**")
        
        strengths = [criterion for criterion, score in result.scores.items() if score >= 8.0]
        
        if 'accuracy' in strengths and 'completeness' in strengths:
            st.write("  • Best for detailed, factual investment analysis")
        elif 'clarity' in strengths and 'helpfulness' in strengths:
            st.write("  • Ideal for client-facing communications")
        elif 'professionalism' in strengths:
            st.write("  • Suitable for formal investment reports")
        else:
            st.write("  • General-purpose investment advisory tasks")