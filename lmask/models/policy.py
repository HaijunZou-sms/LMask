from .env_embeddings import (
    TSPDLInitEmbedding,
    TSPDLContext,
    TSPDLRIEContext,
    TSPTWInitEmbedding,
    TSPTWContext,
    TSPTWRIEContext,
)
from .modules import ReldAttention
from rl4co.models import AttentionModelPolicy
from rl4co.models.zoo.am.decoder import AttentionModelDecoder
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding


class TSPDLPolicy(AttentionModelPolicy):
    def __init__(
        self,
        embed_dim: int = 128,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        env_name: str = "tsptw",
        use_graph_context: bool = False,
        init_embedding: TSPDLInitEmbedding = None,
        context_embedding: TSPDLContext = None,
        decoder_class: str = None,
        **kwargs,
    ):
        if init_embedding is None:
            init_embedding = TSPDLInitEmbedding(embed_dim=embed_dim)
        if context_embedding is None:
            context_embedding = TSPDLContext(embed_dim=embed_dim)

        dynamic_embedding = StaticEmbedding(embed_dim=embed_dim)

        if decoder_class == "reld":
            pointer = ReldAttention(embed_dim=embed_dim, num_heads=num_heads)
            decoder = AttentionModelDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                env_name=env_name,
                context_embedding=context_embedding,
                dynamic_embedding=dynamic_embedding,
                use_graph_context=use_graph_context,
                pointer=pointer,
            )
        else:
            decoder = None

        super(TSPDLPolicy, self).__init__(
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            env_name=env_name,
            use_graph_context=use_graph_context,
            init_embedding=init_embedding,
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            decoder=decoder,
            **kwargs,
        )


class TSPDLRIEPolicy(AttentionModelPolicy):
    def __init__(
        self,
        num_intensity_classes: int = 5,
        embed_dim: int = 128,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        env_name: str = "tsptw",
        use_graph_context: bool = False,
        init_embedding: TSPDLInitEmbedding = None,
        context_embedding: TSPDLRIEContext = None,
        decoder_class: str = None,
        **kwargs,
    ):
        if init_embedding is None:
            init_embedding = TSPDLInitEmbedding(embed_dim=embed_dim)
        if context_embedding is None:
            context_embedding = TSPDLRIEContext(embed_dim=embed_dim, num_intensity_classes=num_intensity_classes)

        dynamic_embedding = StaticEmbedding(embed_dim=embed_dim)

        if decoder_class == "reld":
            pointer = ReldAttention(embed_dim=embed_dim, num_heads=num_heads)
            decoder = AttentionModelDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                env_name=env_name,
                context_embedding=context_embedding,
                dynamic_embedding=dynamic_embedding,
                use_graph_context=use_graph_context,
                pointer=pointer,
            )
        else:
            decoder = None

        super(TSPDLRIEPolicy, self).__init__(
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            env_name=env_name,
            use_graph_context=use_graph_context,
            init_embedding=init_embedding,
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            decoder=decoder,
            **kwargs,
        )


class TSPTWPolicy(AttentionModelPolicy):
    def __init__(
        self,
        embed_dim: int = 128,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        env_name: str = "tsptw",
        use_graph_context: bool = False,
        init_embedding: TSPTWInitEmbedding = None,
        context_embedding: TSPTWContext = None,
        decoder_class: str = None,
        **kwargs,
    ):
        if init_embedding is None:
            init_embedding = TSPTWInitEmbedding(embed_dim=embed_dim)
        if context_embedding is None:
            context_embedding = TSPTWContext(embed_dim=embed_dim)

        dynamic_embedding = StaticEmbedding(embed_dim=embed_dim)

        if decoder_class == "reld":
            pointer = ReldAttention(embed_dim=embed_dim, num_heads=num_heads)
            decoder = AttentionModelDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                env_name=env_name,
                context_embedding=context_embedding,
                dynamic_embedding=dynamic_embedding,
                use_graph_context=use_graph_context,
                pointer=pointer,
            )
        else:
            decoder = None

        super(TSPTWPolicy, self).__init__(
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            env_name=env_name,
            use_graph_context=use_graph_context,
            init_embedding=init_embedding,
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            decoder=decoder,
            **kwargs,
        )


class TSPTWRIEPolicy(AttentionModelPolicy):
    def __init__(
        self,
        num_intensity_classes: int = 5,
        embed_dim: int = 128,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        env_name: str = "tsptw",
        use_graph_context: bool = False,
        init_embedding: TSPTWInitEmbedding = None,
        context_embedding: TSPTWRIEContext = None,
        decoder_class: str = None,
        **kwargs,
    ):
        if init_embedding is None:
            init_embedding = TSPTWInitEmbedding(embed_dim=embed_dim)
        if context_embedding is None:
            context_embedding = TSPTWRIEContext(embed_dim=embed_dim, num_intensity_classes=num_intensity_classes)
        dynamic_embedding = StaticEmbedding(embed_dim=embed_dim)

        if decoder_class == "reld":
            pointer = ReldAttention(embed_dim=embed_dim, num_heads=num_heads)
            decoder = AttentionModelDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                env_name=env_name,
                context_embedding=context_embedding,
                dynamic_embedding=dynamic_embedding,
                use_graph_context=use_graph_context,
                pointer=pointer,
            )
        else:
            decoder = None
        super(TSPTWRIEPolicy, self).__init__(
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            env_name=env_name,
            use_graph_context=use_graph_context,
            init_embedding=init_embedding,
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            decoder=decoder,
            **kwargs,
        )
