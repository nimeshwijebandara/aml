import os
import warnings
import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")



# PAGE CONFIGURATION


st.set_page_config(
    page_title="Real-Time Smart Traffic Routing",
    page_icon="🚦",
    layout="wide",
)



# CONSTANTS


MODEL_PATH = "models/gcn_model.pth"

TRAFFIC_LABELS = {
    0: "Low Traffic",
    1: "Medium Traffic",
    2: "High Traffic",
}



# GCN MODEL


class GCN(torch.nn.Module):

    def __init__(
        self,
        input_dim=5,
        hidden_dim=64,
        output_dim=3,
    ):
        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            hidden_dim,
        )

        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim,
        )

        self.conv3 = GCNConv(
            hidden_dim,
            output_dim,
        )

    def forward(
        self,
        x,
        edge_index,
    ):

        x = self.conv1(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = self.conv3(
            x,
            edge_index,
        )

        return x



# SESSION STATE


if "model" not in st.session_state:
    st.session_state.model = None

if "data" not in st.session_state:
    st.session_state.data = None

if "predictions" not in st.session_state:
    st.session_state.predictions = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None

if "evaluation" not in st.session_state:
    st.session_state.evaluation = None

if "feature_columns" not in st.session_state:
    st.session_state.feature_columns = None



# TITLE


st.title(
    "🚦 Real-Time Smart Traffic Routing"
)

st.write(
    "Graph Neural Network Based Traffic Prediction "
    "with Dynamic Route Optimization"
)



# SIDEBAR


st.sidebar.header(
    "Project Information"
)

st.sidebar.write(
    "🔹 Graph Convolutional Network (GCN)"
)

st.sidebar.write(
    "🔹 Dijkstra Algorithm"
)

st.sidebar.write(
    "🔹 A* Algorithm"
)

st.sidebar.write(
    "🔹 Dynamic Edge Weighting"
)

st.sidebar.write(
    "🔹 Real-Time Traffic Prediction"
)

st.sidebar.divider()

st.sidebar.header(
    "Model Information"
)

st.sidebar.write(
    f"Model: `{MODEL_PATH}`"
)

st.sidebar.write(
    "Classes:"
)

st.sidebar.write(
    "0 → Low Traffic"
)

st.sidebar.write(
    "1 → Medium Traffic"
)

st.sidebar.write(
    "2 → High Traffic"
)



# DATASET SECTION


st.header(
    "📊 Dataset"
)

uploaded = st.file_uploader(
    "Upload TaxiGPS Dataset",
    type=["csv"],
)

df = None


if uploaded is not None:

    try:

        df = pd.read_csv(
            uploaded
        )

        st.success(
            "✅ Dataset Loaded Successfully"
        )

        st.subheader(
            "Dataset Preview"
        )

        st.dataframe(
            df.head(10),
            use_container_width=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Rows",
                f"{df.shape[0]:,}",
            )

        with col2:

            st.metric(
                "Columns",
                df.shape[1],
            )

        with col3:

            st.metric(
                "Missing Values",
                f"{df.isnull().sum().sum():,}",
            )

        with st.expander(
            "View Dataset Columns"
        ):

            st.write(
                list(df.columns)
            )

        if "Traffic_Level" in df.columns:

            st.subheader(
                "🚦 Traffic Distribution"
            )

            traffic_counts = (
                df["Traffic_Level"]
                .value_counts()
            )

            st.bar_chart(
                traffic_counts
            )

        else:

            st.warning(
                "The dataset does not contain "
                "'Traffic_Level'."
            )

    except Exception as e:

        st.error(
            f"❌ Dataset Loading Error: {e}"
        )

# FEATURE PREPARATION

st.header(
    "⚙️ Data Preparation"
)

def prepare_features(
    dataframe
):

    """
    Prepare numerical features for GCN.
    """

    numeric_columns = (
        dataframe
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    # Remove target
    if "Traffic_Level" in numeric_columns:

        numeric_columns.remove(
            "Traffic_Level"
        )

    if len(numeric_columns) == 0:

        raise ValueError(
            "No numerical features were found "
            "in the dataset."
        )

    features = dataframe[
        numeric_columns
    ].copy()

    # Replace infinity
    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Fill missing values
    features = features.fillna(
        features.median(
            numeric_only=True
        )
    )

    features = features.fillna(
        0
    )

    return (
        features,
        numeric_columns,
    )



# GRAPH CREATION

def create_graph(
    num_nodes
):

    """ Creates a bidirectional graph connecting consecutive records. """

    edge_list = []

    if num_nodes > 1:

        for i in range(
            num_nodes - 1
        ):

            edge_list.append(
                [i, i + 1]
            )

            edge_list.append(
                [i + 1, i]
            )

    if len(edge_list) == 0:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

    else:

        edge_index = torch.tensor(
            edge_list,
            dtype=torch.long,
        ).t().contiguous()

    return edge_index


# MODEL SECTION

st.header(
    "🤖 GCN Model"
)


if not os.path.exists(
    MODEL_PATH
):

    st.error(
        f"❌ Model file not found:\n\n"
        f"`{MODEL_PATH}`"
    )

else:

    st.success(
        f"✅ Model file found: `{MODEL_PATH}`"
    )


# MODEL CONFIGURATION

st.subheader(
    "GCN Configuration"
)

col1, col2, col3 = st.columns(3)


with col1:

    hidden_dim = st.number_input(
        "Hidden Dimension",
        min_value=1,
        max_value=512,
        value=64,
        step=1,
    )


with col2:

    output_dim = st.number_input(
        "Output Classes",
        min_value=2,
        max_value=20,
        value=3,
        step=1,
    )


with col3:

    st.write(
        "Traffic Classes"
    )

    st.write(
        "Low / Medium / High"
    )



# LOAD MODEL BUTTON

if st.button(
    "🔄 Load GCN Model",
    use_container_width=True,
):

    if not os.path.exists(
        MODEL_PATH
    ):

        st.error(
            "❌ Model file does not exist."
        )

    elif df is None:

        st.warning(
            "Please upload the TaxiGPS dataset first."
        )

    else:

        try:

            
            # Prepare features
            

            features, feature_columns = (
                prepare_features(df)
            )

            input_dim = len(
                feature_columns
            )

            st.info(
                f"Detected {input_dim} numerical "
                f"input features."
            )

            st.write(
                "Features used by model:"
            )

            st.code(
                ", ".join(
                    feature_columns
                )
            )

            
            # Create model
            

            model = GCN(
                input_dim=input_dim,
                hidden_dim=int(
                    hidden_dim
                ),
                output_dim=int(
                    output_dim
                ),
            )

            
            # Load model checkpoint
            

            checkpoint = torch.load(
                MODEL_PATH,
                map_location="cpu",
                weights_only=False,
            )

            
            # Find state dictionary
            

            if isinstance(
                checkpoint,
                dict
            ):

                if (
                    "model_state_dict"
                    in checkpoint
                ):

                    state_dict = (
                        checkpoint[
                            "model_state_dict"
                        ]
                    )

                elif (
                    "state_dict"
                    in checkpoint
                ):

                    state_dict = (
                        checkpoint[
                            "state_dict"
                        ]
                    )

                else:

                    state_dict = checkpoint

            else:

                state_dict = checkpoint

            
            # Remove module. prefix
            cleaned_state_dict = {}

            for key, value in (
                state_dict.items()
            ):

                new_key = key

                if new_key.startswith(
                    "module."
                ):

                    new_key = new_key[
                        7:
                    ]

                cleaned_state_dict[
                    new_key
                ] = value

            
            # Load weights
           
            model.load_state_dict(
                cleaned_state_dict,
                strict=True,
            )

            model.eval()

            
            # Save model
           
            st.session_state.model = (
                model
            )

            st.session_state.feature_columns = (
                feature_columns
            )

           
            # Create graph
           
            edge_index = create_graph(
                len(df)
            )

            # Create feature tensor
            
            x = torch.tensor(
                features.values,
                dtype=torch.float32,
            )

            # PyTorch Geometric Data
            
            data = Data(
                x=x,
                edge_index=edge_index,
            )

            st.session_state.data = (
                data
            )

            # Clear previous predictions
            st.session_state.predictions = (
                None
            )

            st.session_state.probabilities = (
                None
            )

            # Success
            
            st.success(
                "✅ GCN Model Loaded Successfully!"
            )

            col1, col2, col3 = (
                st.columns(3)
            )

            with col1:

                st.metric(
                    "Input Features",
                    input_dim,
                )

            with col2:

                st.metric(
                    "Hidden Dimension",
                    int(hidden_dim),
                )

            with col3:

                st.metric(
                    "Output Classes",
                    int(output_dim),
                )

            st.info(
                f"Graph created with "
                f"{len(df):,} nodes and "
                f"{edge_index.shape[1]:,} edges."
            )

        except RuntimeError as e:

            st.error(
                "❌ Model Architecture Mismatch"
            )

            st.code(
                str(e)
            )

            st.info( """The trained model architecture does not match the architecture configured in this application."""
            )

        except Exception as e:

            st.error(
                f"❌ Model Loading Error: {e}"
            )

# MODEL STATUS

if (
    st.session_state.model
    is not None
):

    st.success(
        "🟢 GCN Model Status: Loaded"
    )

else:

    st.warning(
        "🔴 GCN Model Status: Not Loaded"
    )

# TRAFFIC PREDICTION

st.header(
    "🔮 Traffic Prediction"
)


if st.button(
    "🚦 Predict Traffic",
    use_container_width=True,
):

    try:

         # Check model
        
        model = (
            st.session_state.get(
                "model"
            )
        )

        if model is None:

            st.warning(
                "Please load the GCN model first."
            )

            st.stop()

        # Check graph
        data = (
            st.session_state.get(
                "data"
            )
        )

        if data is None:

            st.warning(
                "Dataset graph has not been created."
            )

            st.stop()

        # CPU

        device = torch.device(
            "cpu"
        )

        model = model.to(
            device
        )

        model.eval()

        x = data.x.to(
            device
        )

        edge_index = (
            data.edge_index.to(
                device
            )
        )

 
        # Debug information
      
        st.info(
            f"Input shape: {tuple(x.shape)} | "
            f"Edge shape: {tuple(edge_index.shape)}"
        )

         # Validate input
   
        if x.ndim != 2:

            raise ValueError(
                f"Invalid input shape: "
                f"{tuple(x.shape)}"
            )

        if edge_index.ndim != 2:

            raise ValueError(
                f"Invalid edge_index shape: "
                f"{tuple(edge_index.shape)}"
            )

        if edge_index.shape[0] != 2:

            raise ValueError(
                "edge_index must have shape "
                "[2, number_of_edges]."
            )

        # Prediction
  
        with torch.no_grad():

            output = model(
                x,
                edge_index
            )

 
        # Validate output
   
        if output.ndim != 2:

            raise ValueError(
                f"Unexpected model output shape: "
                f"{tuple(output.shape)}"
            )

          # Probability
   
        probabilities = torch.softmax(
            output,
            dim=1
        )

  
        # Predicted class
  
        predictions = torch.argmax(
            probabilities,
            dim=1
        )

         # CPU
 
        predictions = (
            predictions.cpu()
        )

        probabilities = (
            probabilities.cpu()
        )

         # Save
 

        st.session_state.predictions = (
            predictions
        )

        st.session_state.probabilities = (
            probabilities
        )

        # Success
        
        st.success(
            "✅ Traffic Prediction Completed Successfully!"
        )

        # Prediction values
      
        prediction_values = (
            predictions.numpy()
        )

         # Count traffic
        
        low_count = int(
            (
                prediction_values == 0
            ).sum()
        )

        medium_count = int(
            (
                prediction_values == 1
            ).sum()
        )

        high_count = int(
            (
                prediction_values == 2
            ).sum()
        )

         # Traffic Summary
        
        st.subheader(
            "🚦 Traffic Summary"
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:

            st.metric(
                "🟢 Low Traffic",
                low_count
            )

        with col2:

            st.metric(
                "🟡 Medium Traffic",
                medium_count
            )

        with col3:

            st.metric(
                "🔴 High Traffic",
                high_count
            )

        # Prediction Results

        st.subheader(
            "📊 Prediction Results"
        )

        rows = []

        for i in range(
            min(
                20,
                len(prediction_values)
            )
        ):

            class_id = int(
                prediction_values[i]
            )

            confidence = float(
                probabilities[i]
                .max()
                .item()
                * 100
            )

            rows.append(
                {
                    "Node": i,

                    "Predicted Class": (
                        class_id
                    ),

                    "Traffic Level": (
                        TRAFFIC_LABELS.get(
                            class_id,
                            "Unknown"
                        )
                    ),

                    "Confidence": (
                        f"{confidence:.2f}%"
                    ),
                }
            )

        result_df = pd.DataFrame(
            rows
        )

        st.dataframe(
            result_df,
            use_container_width=True
        )

        # Distribution

        st.subheader(
            "📈 Prediction Distribution"
        )

        distribution = pd.Series(
            {
                "Low Traffic": low_count,
                "Medium Traffic": medium_count,
                "High Traffic": high_count,
            }
        )

        st.bar_chart(
            distribution
        )

    except Exception as e:

        st.error(
            "❌ Prediction Error"
        )

        st.exception(e)


# DISPLAY ALL PREDICTIONS
if (
    st.session_state.predictions
    is not None
):

    st.subheader(
        "📋 All Prediction Results"
    )

    prediction_values = (
        st.session_state
        .predictions
        .numpy()
    )

    result_df = pd.DataFrame(
        {
            "Node": range(
                len(prediction_values)
            ),

            "Predicted Class": (
                prediction_values
            ),

            "Traffic Level": [
                TRAFFIC_LABELS.get(
                    int(value),
                    "Unknown"
                )
                for value
                in prediction_values
            ],
        }
    )

    st.dataframe(
        result_df.head(100),
        use_container_width=True,
    )



# MODEL EVALUATION

st.header(
    "📈 Model Evaluation"
)


if (
    st.session_state.predictions
    is None
):

    st.info(
        "Run traffic prediction first."
    )

elif df is None:

    st.info(
        "Upload a dataset to evaluate the model."
    )

elif "Traffic_Level" not in df.columns:

    st.info(
        "Evaluation requires a "
        "`Traffic_Level` column."
    )

else:

    try:

     
        # Predictions
       
        y_pred = (
            st.session_state
            .predictions
            .numpy()
        )

        
        # Actual target
        
        y_true_raw = (
            df["Traffic_Level"]
            .values
        )

        # Label mapping
       
        label_mapping = {
            "Low": 0,
            "low": 0,
            "LOW": 0,

            "Medium": 1,
            "medium": 1,
            "MEDIUM": 1,

            "High": 2,
            "high": 2,
            "HIGH": 2,
        }

        if (
            y_true_raw.dtype.kind
            in "OUS"
        ):

            y_true = np.array(
                [
                    label_mapping.get(
                        str(value).strip(),
                        -1
                    )
                    for value
                    in y_true_raw
                ]
            )

        else:

            y_true = (
                y_true_raw.astype(
                    int
                )
            )

         # Match lengths
        
        min_length = min(
            len(y_true),
            len(y_pred)
        )

        y_true = (
            y_true[:min_length]
        )

        y_pred = (
            y_pred[:min_length]
        )

        
        # Remove invalid labels
       
        valid_mask = (
            y_true >= 0
        )

        y_true = (
            y_true[valid_mask]
        )

        y_pred = (
            y_pred[valid_mask]
        )

        if len(y_true) == 0:

            raise ValueError(
                "No valid target labels were found."
            )

        
        # Metrics
       

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        precision = precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

       
        # Metrics display
       

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        with col1:

            st.metric(
                "Accuracy",
                f"{accuracy * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Precision",
                f"{precision * 100:.2f}%"
            )

        with col3:

            st.metric(
                "Recall",
                f"{recall * 100:.2f}%"
            )

        with col4:

            st.metric(
                "F1 Score",
                f"{f1 * 100:.2f}%"
            )

       
        # ROC-AUC
       
        probabilities = (
            st.session_state
            .probabilities
            .numpy()
        )

        probabilities = (
            probabilities[
                :min_length
            ]
        )

        probabilities = (
            probabilities[
                valid_mask
            ]
        )

        try:

            roc = roc_auc_score(
                y_true,
                probabilities,
                multi_class="ovr",
                average="weighted"
            )

            st.metric(
                "ROC-AUC",
                f"{roc:.3f}"
            )

        except Exception as e:

            st.warning(
                f"ROC-AUC could not be calculated: {e}"
            )

       # Confusion Matrix
         
        st.subheader(
            "Confusion Matrix"
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[
                0,
                1,
                2
            ]
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual Low",
                "Actual Medium",
                "Actual High",
            ],
            columns=[
                "Predicted Low",
                "Predicted Medium",
                "Predicted High",
            ],
        )

        st.dataframe(
            cm_df,
            use_container_width=True,
        )

        
        # Classification Report
        
        st.subheader(
            "Classification Report"
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=[
                0,
                1,
                2
            ],
            target_names=[
                "Low Traffic",
                "Medium Traffic",
                "High Traffic",
            ],
            zero_division=0,
        )

        st.code(
            report
        )

    except Exception as e:

        st.error(
            f"❌ Evaluation Error: {e}"
        )

# DYNAMIC ROUTE OPTIMIZATION
st.header(
    "🛣️ Dynamic Route Optimization"
)

col1, col2 = st.columns(2)

with col1:

    start_node = st.number_input(
        "Start Node",
        min_value=0,
        value=0,
        step=1,
    )

with col2:

    end_node = st.number_input(
        "Destination Node",
        min_value=1,
        value=10,
        step=1,
    )


# CREATE DYNAMIC GRAPH



def create_dynamic_graph(
    dataframe,
    predictions,
):

    G = nx.Graph()

    num_nodes = len(
        dataframe
    )

    # Add nodes
    for node in range(
        num_nodes
    ):

        G.add_node(
            node
        )

    # Add edges
    for i in range(
        num_nodes - 1
    ):

        traffic_class = int(
            predictions[i]
        )

        # Traffic multiplier
        if traffic_class == 0:

            traffic_multiplier = 1.0

        elif traffic_class == 1:

            traffic_multiplier = 1.5

        else:

            traffic_multiplier = 3.0

        base_distance = 1.0

        dynamic_weight = (
            base_distance
            * traffic_multiplier
        )

        G.add_edge(
            i,
            i + 1,
            weight=dynamic_weight,
            traffic=traffic_class,
        )

    return G



# OPTIMIZE ROUTE

if st.button(
    "🚗 Optimize Route",
    use_container_width=True,
):

    if df is None:

        st.warning(
            "Please upload the dataset."
        )

    elif (
        st.session_state.predictions
        is None
    ):

        st.warning(
            "Please run traffic prediction first."
        )

    else:

        try:

            predictions = (
                st.session_state
                .predictions
                .numpy()
            )

          
            # Validate nodes
           
            if (
                start_node >= len(df)
                or end_node >= len(df)
            ):

                st.error(
                    "Start or destination node "
                    "is outside the dataset."
                )

            elif (
                start_node == end_node
            ):

                st.warning(
                    "Start and destination "
                    "cannot be the same."
                )

            else:

                # Create graph
              
                G = create_dynamic_graph(
                    df,
                    predictions
                )

                # Dijkstra
                
                dijkstra_route = (
                    nx.shortest_path(
                        G,
                        source=int(
                            start_node
                        ),
                        target=int(
                            end_node
                        ),
                        weight="weight",
                    )
                )

                dijkstra_cost = (
                    nx.shortest_path_length(
                        G,
                        source=int(
                            start_node
                        ),
                        target=int(
                            end_node
                        ),
                        weight="weight",
                    )
                )

                 # A*
                
                astar_route = (
                    nx.astar_path(
                        G,
                        source=int(
                            start_node
                        ),
                        target=int(
                            end_node
                        ),
                        weight="weight",
                    )
                )

                astar_cost = (
                    nx.astar_path_length(
                        G,
                        source=int(
                            start_node
                        ),
                        target=int(
                            end_node
                        ),
                        weight="weight",
                    )
                )

                 # Success
                

                st.success(
                    "✅ Optimal Route Generated!"
                )

                col1, col2 = (
                    st.columns(2)
                )

                  # Dijkstra result
                

                with col1:

                    st.subheader(
                        "🔵 Dijkstra Route"
                    )

                    st.write(
                        " → ".join(
                            map(
                                str,
                                dijkstra_route
                            )
                        )
                    )

                    st.metric(
                        "Route Cost",
                        f"{dijkstra_cost:.2f}"
                    )

                    st.metric(
                        "Number of Nodes",
                        len(
                            dijkstra_route
                        )
                    )

              
                # A* result
                with col2:

                    st.subheader(
                        "🟢 A* Route"
                    )

                    st.write(
                        " → ".join(
                            map(
                                str,
                                astar_route
                            )
                        )
                    )

                    st.metric(
                        "Route Cost",
                        f"{astar_cost:.2f}"
                    )

                    st.metric(
                        "Number of Nodes",
                        len(
                            astar_route
                        )
                    )

                
                # Comparison
                

                st.subheader(
                    "📊 Route Comparison"
                )

                comparison = pd.DataFrame(
                    {
                        "Algorithm": [
                            "Dijkstra",
                            "A*",
                        ],

                        "Route Cost": [
                            dijkstra_cost,
                            astar_cost,
                        ],

                        "Number of Nodes": [
                            len(
                                dijkstra_route
                            ),
                            len(
                                astar_route
                            ),
                        ],
                    }
                )

                st.dataframe(
                    comparison,
                    use_container_width=True,
                )

        except nx.NetworkXNoPath:

            st.error(
                "❌ No route exists between "
                "the selected nodes."
            )

        except Exception as e:

            st.error(
                f"❌ Route Optimization Error: {e}"
            )



# ABOUT


st.header(
    "ℹ️ About"
)

st.write(
    """
This application demonstrates a Real-Time Smart Traffic Routing System using Graph Convolutional Networks (GCN) with Dynamic Route Optimization.

The system combines:

• Graph Convolutional Network (GCN)
• Real-time traffic prediction
• Dynamic edge weighting
• Dijkstra shortest-path algorithm
• A* pathfinding algorithm
• Traffic-aware route optimization

Traffic levels:

🟢 Low Traffic
🟡 Medium Traffic
🔴 High Traffic
"""
)



# FOOTER


st.divider()

st.caption(
    "Real-Time Smart Traffic Routing System "
    "using GCN and Dynamic Route Optimization"
)
