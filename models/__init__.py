def get_model(args):
    model_name = args.model_name.lower()

    if model_name == "ncm" or model_name == "aper":
        from .ncm import NCM
        return NCM(device=args.device)

    elif model_name == "slda":
        from .slda import SLDA
        return SLDA(reg=args.reg, device=args.device)

    elif model_name == "klda":
        from .klda import KLDA
        return KLDA(D=args.D, gamma=args.gamma, reg=args.reg, seed=args.seed, device=args.device)

    elif model_name == "gacl":
        from .gacl import GACL
        return GACL(D=args.D, reg=args.reg, seed=args.seed, device=args.device)

    elif model_name == "anacp":
        from .anacp import AnaCP
        return AnaCP(D=args.D, reg=args.reg, num_heads=args.num_heads, seed=args.seed,
                     device=args.device, samples_per_class=args.samples_per_class, shared_cov=args.shared_cov)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
