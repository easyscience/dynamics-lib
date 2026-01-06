# The current SampleModel will be renamed to ComponentCollection or something similar. It will take a list (perhaps sc.array) of Q among other things.

# The SampleModel will allow the user to append DiffusionModels to a list of DiffusionModels and append ModelComponents to a ComponentCollection.

# There will also be a list of ComponentCollections, where each is a copy of the ComponentCollection that the user supplied. The user will also be allowed to work directly with this list. The list is the same length as Q; each ComponentCollection corresponds to a single Q.

# Behind the scenes, it will have a list of ComponentCollection, which contains all the user supplied ComponentCollections.

# The DiffusionModel will also be able to generate components. It may be best to keep them in a separate list of ComponentCollections, just to make sure they don't accidentally get overwritten or changed by the user. It should be possible to append a DiffusionModel without actually generating the components it contains., Fitting entire diffusion models is very difficult until you have a very good understanding of your data, and can take very long - in preliminary tests, fitting sequentially took about 3 seconds, and a DiffusionModel from an ok, but not great, starting point took 15 minutes.

# Perhaps it will have an explicit generate_diffusion_model_components or some such.

# It should have a calculate and plot method to plot the model of the scattering of the sample before convolution. I suppose that they could take two lists of ComponentCollections and make a single ComponentCollection out of them?

# It will have an optional Temperature, which when not None will include detailed balance calculations.

# It will eventually also have to support taking a list of temperatures and allow models to vary as function of temperature - not entirely sure how that will work. Perhaps we will actually have a list of list of ComponentCollection, where one is over temperature, and the other is over Q. Let's deal with that when we get there.

# SampleModel will inherit from SampleModelBase or something similar.
