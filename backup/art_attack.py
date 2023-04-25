def execute_attack(config, attack_name, x, y, z, classifier, hybrid_classifier, autoencoder_model, kwargs, conditionals):
    result = {}
    ## This "all" option is old, not modified according to
    ## the else block below. Modify it and use, if needed.
    if attack_name == "all":
        del ATTACK_MAPPINGS["all"]
        for name, attack_name in ATTACK_MAPPINGS.items():
            result[name] = {}
            print(f"Running {name} attack!!!!!")
            attack = attack_name(classifier)
            x_test_adv_np = attack.generate(x=x[1])
            predictions = classifier.predict(x_test_adv_np)
            accuracy = np.sum(np.argmax(predictions, axis=-1) == y[1]) / len(y[1])
            result[name]["original"] = accuracy
            result[name]["x_test_adv_np"] = x_test_adv_np

            # calculate noise
            x_test_noise = x_test_adv_np - x[1]
            result[name]["x_test_noise"] = x_test_noise
            print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

            modified_attack = attack_name(hybrid_classifier)
            z_test_adv_np = modified_attack.generate(x=z[1])

            # Step 7: Evaluate the ART classifier on adversarial test examples
            predictions = hybrid_classifier.predict(z_test_adv_np)
            accuracy = np.sum(np.argmax(predictions, axis=-1) == y[1]) / len(y[1])
            result[name]["modified"] = accuracy
            result[name]["z_test_adv_np"] = z_test_adv_np
            xx_test_adv = autoencoder_model.decoder(torch.Tensor(z_test_adv_np).to(config["device"]))
            xx_test     = autoencoder_model.decoder(torch.Tensor(z[1]).to(config["device"]))

            # calculate noise
            xx_test_noise = xx_test_adv - xx_test
            result[name]["xx_test_noise"] = xx_test_noise.cpu().detach().numpy()
            result[name]["xx_test_adv_np"] = xx_test_adv.cpu().detach().numpy()
            print("Accuracy on adversarial test examples(Modified): {}%".format(accuracy * 100))

            hybrid_noise = result[name]["xx_test_noise"] + 0.1 * x_test_noise
            hybrid_x_np = x[1] + hybrid_noise
            result[name]["hybrid_x_np"] = hybrid_x_np
            result[name]["hybrid_noise"] = hybrid_noise

            predictions = classifier.predict(hybrid_x_np)
            accuracy = np.sum(np.argmax(predictions, axis=-1) == y[1]) / len(y[1])
            print("Accuracy on adversarial test examples(Hybrid): {}%".format(accuracy * 100))
            result[name]["hybrid"] = accuracy
    else:
        name = attack_name.__name__
        result[name] = {}
        
        # ------------------------------------------------- #
        # ---------------- Original Attack ---------------- #
        # ------------------------------------------------- #
        if conditionals["calculate_original"]:
            attack = attack_name(classifier, **kwargs)
            start = time.time()
            x_adv = attack.generate(x=x[1])
            orig_time = time.time() - start
            predictions = classifier.predict(x_adv)
            x_adv_acc = np.sum(np.argmax(predictions, axis=-1) == y[1]) / len(y[1])

            result[name]["x_adv"] = x_adv
            result[name]["x_adv_acc"] = x_adv_acc

            # calculate noise
            delta_x = x_adv - x[1]
            result[name]["delta_x"] = delta_x
            accuracy = np.sum(np.argmax(predictions, axis=-1) == y[1]) / len(y[1])
            # print("Robust accuracy of original adversarial attack: {}%".format(accuracy * 100))

        # ------------------------------------------------- #
        # ---------------- Modified Attack ---------------- #
        # ------------------------------------------------- #
        # print(**kwargs)
        modified_attack = attack_name(hybrid_classifier, **kwargs)
        if conditionals["is_class_constrained"]:
            start = time.time()
            z_adv = modified_attack.generate(x=z[1], mask=generate_mask(
                latent_dim=int(config["latent_shape"]),
                n_classes=config["miscs"]["nb_classes"],
                labels=y[1]))
            modf_time = time.time() - start
        else:
            start = time.time()
            z_adv = modified_attack.generate(x=z[1])
            modf_time = time.time() - start

        # calculate noise
        autoencoder_model = autoencoder_model.to(config["device"])
        x_hat_adv   = autoencoder_model.decoder(torch.Tensor(z_adv).to(config["device"]))
        x_hat       = autoencoder_model.decoder(torch.Tensor(z[1]).to(config["device"]))
        delta_x_hat  = x_hat_adv - x_hat

        # modified attack
        modf_x_adv   = x[1] + delta_x_hat.cpu().detach().numpy()
        predictions = classifier.predict(modf_x_adv)
        modf_x_adv_acc = np.sum(np.argmax(predictions, axis=-1) == y[1]) / len(y[1])

        result[name]["modf_x_adv"] = modf_x_adv
        result[name]["modf_x_adv_acc"] = modf_x_adv_acc

        # reconstructed attack
        predictions = hybrid_classifier.predict(z_adv)
        x_hat_adv_acc = np.sum(np.argmax(predictions, axis=-1) == y[1]) / len(y[1])

        result[name]["z_adv"] = z_adv
        result[name]["x_hat_adv"] = x_hat_adv.cpu().detach().numpy()
        result[name]["x_hat_adv_acc"] = x_hat_adv_acc
        
        # send combined noise
        result[name]["delta_x_hat"] = delta_x_hat.cpu().detach().numpy()

        result[name]["orig_time"] = orig_time
        result[name]["modf_time"] = modf_time

        # print("Robust accuracy of modified adversarial attack: {}%".format(modf_x_adv_acc * 100))
        # print("Robust accuracy of reconstructed adversarial attack: {}%".format(x_hat_adv_acc * 100))

    return result